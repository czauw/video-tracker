import cv2
import numpy as np
from collections import deque
import time

# --- Parameter Configuration ---
NCC_BASE_THRESHOLD = 0.45
TEMPLATE_UPDATE_CONF_THRESHOLD = 0.80
MAX_TEMPLATES = 10
MAX_LOST_FRAMES = 10
RE_DETECTION_FRAMES = 20
GLOBAL_SEARCH_SCALE_FACTOR = 3.0
PYRAMID_MAX_LEVEL = 3
SEARCH_WINDOW_FACTOR = 2.5
MAX_DISPLAY_DIM = 1024 # Max dimension for display windows (can be adjusted)

# --- ORB Feature Detector and Matcher ---
orb = cv2.ORB_create(nfeatures=500)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# --- Kalman Filter Initialization ---
def init_kalman(bbox, dt):
    kf = cv2.KalmanFilter(6, 4)
    kf.transitionMatrix = np.array([
        [1, 0, 0, 0, dt, 0], [0, 1, 0, 0, 0, dt], [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]
    ], np.float32)
    kf.measurementMatrix = np.array([
        [1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0]
    ], np.float32)
    kf.processNoiseCov = np.diag([1, 1, 1, 1, 1e2, 1e2]).astype(np.float32) * 1e-3
    kf.measurementNoiseCov = np.diag([1, 1, 10, 10]).astype(np.float32) * 1e-1
    x, y, w, h = bbox
    cx, cy = x + w / 2, y + h / 2
    kf.statePost = np.array([cx, cy, w, h, 0, 0], np.float32).reshape(6, 1)
    kf.errorCovPost = np.eye(6, dtype=np.float32) * 0.1
    return kf

# --- Global Motion Estimation ---
def estimate_global_affine(prev_gray, curr_gray, orb_detector, bf_matcher):
    try:
        kp1, des1 = orb_detector.detectAndCompute(prev_gray, None)
        kp2, des2 = orb_detector.detectAndCompute(curr_gray, None)
        if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
            return None, None
        matches = bf_matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        if len(matches) < 6:
            return None, None
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        M, mask = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=5.0)
        if M is None or mask is None or np.sum(mask) < 6:
            return None, None
        return M, mask.ravel() == 1
    except (cv2.error, Exception):
        return None, None

# --- NCC Matching ---
def match_ncc(template, search_img):
    try:
        if template.shape[0] > search_img.shape[0] or template.shape[1] > search_img.shape[1] or \
           template.shape[0] == 0 or template.shape[1] == 0 or \
           search_img.shape[0] == 0 or search_img.shape[1] == 0:
            return -1.0, (0, 0)
        res = cv2.matchTemplate(search_img, template, cv2.TM_CCOEFF_NORMED)
        _, maxv, _, maxl = cv2.minMaxLoc(res)
        return maxv, maxl
    except cv2.error:
        return -1.0, (0, 0)

# --- Build Image Pyramid ---
def build_pyramid(image, max_level):
    pyramid = [image]
    for _ in range(max_level):
        try:
            image = cv2.pyrDown(image)
            if image is None or image.shape[0] < 5 or image.shape[1] < 5:
                break
            pyramid.append(image)
        except cv2.error:
            break
    return pyramid

# --- Frame Resizing Utility (New, but similar to previous versions) ---
def resize_frame_if_needed(frame, max_dim):
    """Resizes a frame maintaining aspect ratio if a dimension exceeds max_dim."""
    h_orig, w_orig = frame.shape[:2]
    scale = 1.0
    if max(h_orig, w_orig) > max_dim:
        scale = max_dim / max(h_orig, w_orig)
        frame_resized = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return frame_resized, scale
    return frame, scale

# --- Main Program ---
def main():
    path = input("Enter video path: ")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # --- Initial Frame Selection Loop ---
    selected_frame_for_roi_raw = None # This will store the raw frame chosen by the user
    current_frame_raw = None
    frame_idx_selection = 0
    selection_window_name = "Select Initial Frame (N: Next, S: Select This Frame, Q: Quit)"

    while True:
        ret, frame_capture = cap.read()
        if not ret:
            print("Error: Could not read frame for selection or video ended.")
            cap.release()
            cv2.destroyAllWindows()
            return

        current_frame_raw = frame_capture.copy() # Store the raw frame
        # Resize for display during selection process
        frame_display_selection, _ = resize_frame_if_needed(current_frame_raw, MAX_DISPLAY_DIM)

        # Add text instruction on the frame
        cv2.putText(frame_display_selection, f"Frame: {frame_idx_selection}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame_display_selection, "N: Next, S: Select, Q: Quit", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow(selection_window_name, frame_display_selection)

        key = cv2.waitKey(0) & 0xFF # Wait indefinitely for a key press

        if key == ord('n') or key == ord('N'):
            frame_idx_selection += 1
            continue
        elif key == ord('s') or key == ord('S'):
            selected_frame_for_roi_raw = current_frame_raw # User selected this raw frame
            print(f"Selected frame index {frame_idx_selection} for ROI selection.")
            break
        elif key == ord('q') or key == ord('Q') or key == 27: # ESC key
            print("User quit during frame selection. Exiting.")
            cap.release()
            cv2.destroyAllWindows()
            return
    cv2.destroyWindow(selection_window_name)
    # --- End of Initial Frame Selection Loop ---

    if selected_frame_for_roi_raw is None:
        print("Error: No frame was selected for ROI. Exiting.")
        cap.release()
        return

    # Now, process the selected_frame_for_roi_raw for ROI selection and initialization
    # This frame will be referred to as 'frame_init' henceforth
    frame_init, global_scale_factor = resize_frame_if_needed(selected_frame_for_roi_raw, MAX_DISPLAY_DIM)
    if global_scale_factor != 1.0:
        print(f"Initial frame has been resized for processing. Scale: {global_scale_factor:.2f}. New dims: {frame_init.shape[:2]}")


    roi_selection_window_title = "Select Target ROI (Press Enter/Space to Confirm, C to Cancel)"
    roi = cv2.selectROI(roi_selection_window_title, frame_init, False)
    cv2.destroyWindow(roi_selection_window_title)

    if roi == (0, 0, 0, 0) or roi[2] <= 0 or roi[3] <= 0: # Also check for valid width/height
        print("No valid ROI selected. Exiting.")
        cap.release()
        return
    x, y, w, h = roi
    print(f"Initial ROI (on potentially resized frame): x={x}, y={y}, w={w}, h={h}")

    prev_gray_full = cv2.cvtColor(frame_init, cv2.COLOR_BGR2GRAY)
    prev_gray_processed = cv2.equalizeHist(prev_gray_full)

    init_tmpl = prev_gray_processed[y:y+h, x:x+w].copy()
    if init_tmpl.size == 0:
        print("Error: Initial template is empty. Please select a valid region. Exiting.")
        cap.release()
        return
    print(f"Initial template size: {init_tmpl.shape}")

    templates = deque(maxlen=MAX_TEMPLATES)
    last_time = time.time()
    # Estimate dt using video FPS if available, otherwise default
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    dt_initial = 1.0 / fps_video if fps_video and fps_video > 0 else 1.0 / 30.0
    kf = init_kalman((x, y, w, h), dt_initial)

    traj = []
    lost_counter = 0
    is_re_detecting = False
    last_known_bbox = (x, y, w, h) # In coordinates of frame_init (potentially scaled)

    tracker_display_window_title = "Tracker Output (Press 'q' or ESC to quit)"

    # The video capture 'cap' is now positioned at the frame AFTER the one selected for ROI.
    # This is the correct starting point for the tracking loop.

    while True:
        ret, frame_loop_raw = cap.read()
        if not ret:
            print("Video ended or failed to read next frame.")
            break

        # Apply the SAME scaling to subsequent frames as was applied to the initial frame
        if global_scale_factor != 1.0:
            frame_loop_scaled = cv2.resize(frame_loop_raw, None, fx=global_scale_factor, fy=global_scale_factor, interpolation=cv2.INTER_AREA)
        else:
            frame_loop_scaled = frame_loop_raw.copy() # Use copy if no scaling

        current_time = time.time()
        dt_actual = current_time - last_time
        if dt_actual <= 0:
            dt_actual = 1e-3 # Avoid zero or negative dt
        last_time = current_time

        curr_gray_full = cv2.cvtColor(frame_loop_scaled, cv2.COLOR_BGR2GRAY)
        curr_gray_processed = cv2.equalizeHist(curr_gray_full)

        M, _ = estimate_global_affine(prev_gray_processed, curr_gray_processed, orb, bf)

        kf.transitionMatrix[0, 4] = dt_actual
        kf.transitionMatrix[1, 5] = dt_actual
        prediction = kf.predict()
        pcx, pcy = prediction[0, 0], prediction[1, 0]
        pw, ph = max(5, int(prediction[2, 0])), max(5, int(prediction[3, 0]))

        search_center_x, search_center_y = pcx, pcy
        if M is not None and last_known_bbox is not None:
            last_cx = last_known_bbox[0] + last_known_bbox[2] / 2
            last_cy = last_known_bbox[1] + last_known_bbox[3] / 2
            transformed_center_x = M[0, 0] * last_cx + M[0, 1] * last_cy + M[0, 2]
            transformed_center_y = M[1, 0] * last_cx + M[1, 1] * last_cy + M[1, 2]
            search_center_x = transformed_center_x
            search_center_y = transformed_center_y

        frame_h_proc, frame_w_proc = curr_gray_processed.shape # Dimensions of processed (scaled) frame
        if is_re_detecting:
            search_w_roi = int(last_known_bbox[2] * GLOBAL_SEARCH_SCALE_FACTOR)
            search_h_roi = int(last_known_bbox[3] * GLOBAL_SEARCH_SCALE_FACTOR)
            # Center search on last known position for re-detection
            scx_rd = last_known_bbox[0] + last_known_bbox[2] / 2
            scy_rd = last_known_bbox[1] + last_known_bbox[3] / 2
            sx = max(0, int(scx_rd - search_w_roi / 2))
            sy = max(0, int(scy_rd - search_h_roi / 2))
        else:
            search_w_roi = int(pw * SEARCH_WINDOW_FACTOR)
            search_h_roi = int(ph * SEARCH_WINDOW_FACTOR)
            sx = max(0, int(search_center_x - search_w_roi / 2))
            sy = max(0, int(search_center_y - search_h_roi / 2))

        ex = min(frame_w_proc, sx + search_w_roi)
        ey = min(frame_h_proc, sy + search_h_roi)

        if ex <= sx or ey <= sy: # Check for valid search ROI dimensions
            search_roi_img = np.array([[]], dtype=curr_gray_processed.dtype)
        else:
            search_roi_img = curr_gray_processed[sy:ey, sx:ex]

        best_score = -1.0
        best_bbox_abs = None
        found_in_frame = False

        if search_roi_img.size > 10 and init_tmpl.size > 0 : # Ensure search ROI and init_tmpl are valid
            pyramid = build_pyramid(search_roi_img, PYRAMID_MAX_LEVEL)
            all_templates_to_match = [init_tmpl] + list(templates)

            for tmpl_orig in all_templates_to_match:
                if tmpl_orig.shape[0] == 0 or tmpl_orig.shape[1] == 0: continue

                # --- Corrected Scale Handling Logic ---
                # Match original-sized template against different pyramid levels of the search_roi_img
                # The 'scale_py' refers to how much the search_roi_img was downscaled.
                for level, roi_level_img in enumerate(pyramid):
                    scale_py = 2**level # How much roi_level_img is scaled down from search_roi_img

                    # Template to use for matching (it's the original template from the list)
                    current_template_to_match = tmpl_orig

                    # Ensure template is not larger than the current pyramid level image
                    if current_template_to_match.shape[0] > roi_level_img.shape[0] or \
                       current_template_to_match.shape[1] > roi_level_img.shape[1]:
                        continue

                    score, loc = match_ncc(current_template_to_match, roi_level_img)

                    if score > best_score:
                        best_score = score
                        # loc is top-left in roi_level_img
                        # Convert loc back to coordinates relative to search_roi_img
                        roi_rel_x = int(loc[0] * scale_py)
                        roi_rel_y = int(loc[1] * scale_py)

                        # The size of the match in search_roi_img scale
                        # is the template's size * scale_py
                        match_w = int(current_template_to_match.shape[1] * scale_py)
                        match_h = int(current_template_to_match.shape[0] * scale_py)

                        abs_x = sx + roi_rel_x
                        abs_y = sy + roi_rel_y

                        # Boundary checks for the calculated absolute bbox
                        checked_abs_x = max(0, abs_x)
                        checked_abs_y = max(0, abs_y)
                        checked_w = min(frame_w_proc - checked_abs_x, match_w)
                        checked_h = min(frame_h_proc - checked_abs_y, match_h)

                        if checked_w > 0 and checked_h > 0:
                            best_bbox_abs = (checked_abs_x, checked_abs_y, checked_w, checked_h)
            # --- End of Corrected Scale Handling ---


            current_threshold = NCC_BASE_THRESHOLD
            if is_re_detecting:
                current_threshold *= 0.9

            if best_score >= current_threshold and best_bbox_abs is not None:
                found_in_frame = True
                x_m, y_m, w_m, h_m = best_bbox_abs
                cx_m, cy_m = x_m + w_m / 2, y_m + h_m / 2
                meas = np.array([cx_m, cy_m, w_m, h_m], np.float32).reshape(4, 1)
                kf.correct(meas)

                if best_score > TEMPLATE_UPDATE_CONF_THRESHOLD and not is_re_detecting:
                    if y_m + h_m <= curr_gray_processed.shape[0] and x_m + w_m <= curr_gray_processed.shape[1] and h_m > 0 and w_m > 0:
                        new_patch = curr_gray_processed[y_m : y_m + h_m, x_m : x_m + w_m]
                        if new_patch.size > 0 and init_tmpl.shape[0] > 0 and init_tmpl.shape[1] > 0:
                            try:
                                resized_patch = cv2.resize(new_patch, (init_tmpl.shape[1], init_tmpl.shape[0]), interpolation=cv2.INTER_AREA)
                                templates.append(resized_patch)
                            except cv2.error:
                                pass
                lost_counter = 0
                if is_re_detecting:
                    print("Info: Target re-acquired!")
                is_re_detecting = False
                last_known_bbox = best_bbox_abs
            else:
                found_in_frame = False
                lost_counter += 1
                if not is_re_detecting and lost_counter > MAX_LOST_FRAMES:
                    print(f"Info: Target lost for {lost_counter} frames. Entering re-detection mode...")
                    is_re_detecting = True
                    lost_counter = 0
                elif is_re_detecting and lost_counter > RE_DETECTION_FRAMES:
                    print(f"Info: Re-detection failed after {RE_DETECTION_FRAMES} frames. Target considered lost.")
                    is_re_detecting = False
                    lost_counter = MAX_LOST_FRAMES + 1 # Mark as definitely lost
        else:
             found_in_frame = False
             lost_counter += 1
             if not is_re_detecting and lost_counter > MAX_LOST_FRAMES:
                 is_re_detecting = True; lost_counter = 0
             elif is_re_detecting and lost_counter > RE_DETECTION_FRAMES:
                 is_re_detecting = False; lost_counter = MAX_LOST_FRAMES + 1


        display_frame_output = frame_loop_scaled.copy()
        box_color = (0, 0, 255)
        status_text = "Status: Lost"
        x_draw, y_draw, w_draw, h_draw = 0,0,0,0


        if found_in_frame and best_bbox_abs:
            x_draw, y_draw, w_draw, h_draw = best_bbox_abs
            center_x_traj, center_y_traj = int(x_draw + w_draw / 2), int(y_draw + h_draw / 2)
            if w_draw > 0 and h_draw > 0: traj.append((center_x_traj, center_y_traj))
            box_color = (0, 255, 0)
            status_text = f"Status: Tracking (Score: {best_score:.2f})"
        elif is_re_detecting:
            x_draw, y_draw = int(pcx - pw / 2), int(pcy - ph / 2) # Use Kalman prediction
            w_draw, h_draw = int(pw), int(ph)
            status_text = f"Status: Re-detecting... (Attempt: {lost_counter})"
        else: # Lost
            x_draw, y_draw = int(pcx - pw / 2), int(pcy - ph / 2) # Use Kalman prediction
            w_draw, h_draw = int(pw), int(ph)
            status_text = f"Status: Lost ({lost_counter} frames)"

        # Ensure drawing coordinates are integers and box is valid before drawing
        x_draw, y_draw, w_draw, h_draw = map(int, [x_draw, y_draw, w_draw, h_draw])
        if w_draw > 0 and h_draw > 0:
            # Clip drawing coordinates to be within frame boundaries
            x_c = max(0, x_draw)
            y_c = max(0, y_draw)
            w_c = min(w_draw, display_frame_output.shape[1] - x_c)
            h_c = min(h_draw, display_frame_output.shape[0] - y_c)
            if w_c > 0 and h_c > 0: # Draw only if still valid after clipping
                 cv2.rectangle(display_frame_output, (x_c, y_c), (x_c + w_c, y_c + h_c), box_color, 2)


        max_traj_points_draw = 100
        if len(traj) > 1:
            visible_traj_pts = traj[max(0, len(traj)-max_traj_points_draw):]
            if visible_traj_pts: # Ensure list is not empty
                 pts_traj = np.array(visible_traj_pts, np.int32).reshape(-1, 1, 2)
                 cv2.polylines(display_frame_output, [pts_traj], isClosed=False, color=(255, 0, 0), thickness=2)

        cv2.putText(display_frame_output, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
        cv2.imshow(tracker_display_window_title, display_frame_output)

        prev_gray_full = curr_gray_full.copy()
        prev_gray_processed = curr_gray_processed.copy()

        key = cv2.waitKey(50) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Tracker finished.")

if __name__ == '__main__':
    main()
