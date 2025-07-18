import glob
# import imageio.v2 as imageio
from matplotlib import path as mpl_path
import matplotlib.patheffects as pe
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from microbundlecompute import image_analysis as ia
import numpy as np
from pathlib import Path
from typing import List, Union, Tuple
# from PIL import Image  


def box_to_mask(mask: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Given a mask (for dimensions) and a box. Will return a mask of the inside of the box."""
    p = mpl_path.Path([(box[0, 0], box[0, 1]), (box[1, 0], box[1, 1]), (box[2, 0], box[2, 1]), (box[3, 0], box[3, 1])])
    new_mask = np.zeros(mask.shape)
    for rr in range(0, mask.shape[0]):
        for cc in range(0, mask.shape[1]):
            new_mask[rr, cc] = p.contains_points([(rr, cc)])
    return new_mask


def shrink_box(box: np.ndarray, shrink_row: float = 0.1, shrink_col: float = 0.1) -> np.ndarray:
    """Given a box and shrink factors. Will make the box smaller by the given fraction."""
    r0, r1, c0, c1 = ia.box_to_bound(box)
    r0_new, r1_new = ia.shrink_pair(r0, r1, shrink_row)
    c0_new, c1_new = ia.shrink_pair(c0, c1, shrink_col)
    box = ia.bound_to_box(r0_new, r1_new, c0_new, c1_new)
    return box


def create_sub_domains(
    mask: np.ndarray,
    *,
    pillar_clip_fraction: float = 0.5,
    shrink_row: float = 0.1,
    shrink_col: float = 0.1,
    tile_dim_pix: int = 40,
    num_tile_row: int = 5,
    num_tile_col: int = 3,
    tile_style: int = 1,
    clip_columns: bool = True,
    clip_rows: bool = False,
    manual_sub: bool = False,
    sub_extents: List = None
) -> List:
    """Given a mask and sub-domain parameters. Will return a list of sub-domains define by 4 box coordinates.
    tile_style = 1 will fit as many square tiles of the given tile_dim_pix size in a grid
    tyle_style = 2 will create a num_tile_row x num_tile_col sized grid and adjust the tile_dim_pix as need
    """
    # clip pillars from the mask
    mask_removed_pillars = ia.remove_pillar_region(mask, pillar_clip_fraction,clip_columns, clip_rows)

    if manual_sub:
        r0_user, r1_user, c0_user, c1_user = sub_extents
        user_box = ia.bound_to_box(r0_user, r1_user, c0_user, c1_user)
        first_mask = box_to_mask(mask_removed_pillars, user_box)
        first_box_mask = ia.mask_to_box(first_mask)
        box_mask = shrink_box(first_box_mask, shrink_row, shrink_col)
    else:    
        # compute overall box
        first_box_mask = ia.mask_to_box(mask_removed_pillars)
        box_mask = shrink_box(first_box_mask, shrink_row, shrink_col)
    # tile sub-domains
    r0, r1, c0, c1 = ia.box_to_bound(box_mask)
    if tile_style == 1:
        num_tile_row = int(np.floor((r1 - r0) / tile_dim_pix))
        num_tile_col = int(np.floor((c1 - c0) / tile_dim_pix))
    elif tile_style == 2:
        tile_dim_pix = np.min([np.floor((r1 - r0) / num_tile_row), np.floor((c1 - c0) / num_tile_col)])
    # sub-divide into sub-domains
    shrink_row = 1.0 - num_tile_row * tile_dim_pix / (r1 - r0)
    shrink_col = 1.0 - num_tile_col * tile_dim_pix / (c1 - c0)
    box_mask_grid = shrink_box(box_mask, shrink_row, shrink_col)
    r0_box, _, c0_box, _ = ia.box_to_bound(box_mask_grid)
    tile_box_list = []
    for rr in range(0, num_tile_row):
        for cc in range(0, num_tile_col):
            tile_box = ia.bound_to_box(r0_box + rr * tile_dim_pix, r0_box + (rr + 1) * tile_dim_pix, c0_box + cc * tile_dim_pix, c0_box + (cc + 1) * tile_dim_pix)
            tile_box_list.append(tile_box)
    return tile_box_list, tile_dim_pix, num_tile_row, num_tile_col


def isolate_sub_domain_markers(tracker_row_all: List, tracker_col_all: List, sd_box: np.ndarray) -> List:
    """Given tracker row and column arrays and sub-domain box. Will return markers inside sub-domain."""
    sd_tracker_row_all = []
    sd_tracker_col_all = []
    for kk in range(0, len(tracker_row_all)):
        tracker_row = tracker_row_all[kk]
        tracker_col = tracker_col_all[kk]
        sd_tracker_row = []
        sd_tracker_col = []
        for jj in range(0, tracker_row.shape[0]):
            rr = tracker_row[jj, 0]
            cc = tracker_col[jj, 0]
            if ia.is_in_box(sd_box, rr, cc):
                sd_tracker_row.append(tracker_row[jj, :])
                sd_tracker_col.append(tracker_col[jj, :])
        sd_tracker_row = np.asarray(sd_tracker_row)
        sd_tracker_col = np.asarray(sd_tracker_col)
        sd_tracker_row_all.append(sd_tracker_row)
        sd_tracker_col_all.append(sd_tracker_col)
    return sd_tracker_row_all, sd_tracker_col_all


def compute_F_from_Lambda_mat(Lambda_0: np.ndarray, Lambda_t: np.ndarray) -> np.ndarray:
    """Given Lambda (i.e., vectors connecting fiducial marker positions) matricies in positions 0 and t.
    Lambda matricies are units 2 x number of points.
    Will return the average deformation gradient F."""
    term_1 = np.dot(Lambda_t, np.transpose(Lambda_0))
    #term_2 = np.linalg.inv(np.dot(Lambda_0, np.transpose(Lambda_0)))
    term_2 = np.linalg.pinv(np.dot(Lambda_0, np.transpose(Lambda_0)))
    F = np.dot(term_1, term_2)
    # add in error handling if issues arise here --
    return F


def compute_Lambda_from_pts(row_pos: np.ndarray, col_pos: np.ndarray) -> np.ndarray:
    num_pts = row_pos.shape[0]
    pts_row_col = np.array([row_pos,col_pos])
    ii, jj = np.triu_indices(num_pts, k=1)
    Lambda_mat = pts_row_col[:,ii] - pts_row_col[:,jj]
    return Lambda_mat


def compute_sub_domain_strain(sd_tracker_row: np.ndarray, sd_tracker_col: np.ndarray) -> np.ndarray:
    """Given tracking point positions. Will return F at every frame with the first frame as the reference."""
    sd_F_list = []
    num_frames = sd_tracker_row.shape[1]
    Lambda_0 = compute_Lambda_from_pts(sd_tracker_row[:, 0], sd_tracker_col[:, 0])
    for kk in range(0, num_frames):
        Lambda_t = compute_Lambda_from_pts(sd_tracker_row[:, kk], sd_tracker_col[:, kk])
        F = compute_F_from_Lambda_mat(Lambda_0, Lambda_t)
        sd_F_list.append(F.reshape((-1, 1)))
    sd_F_arr = np.asarray(sd_F_list)[:, :, 0]
    return sd_F_arr


def get_box_center(box: np.ndarray) -> Union[float, int]:
    """Given a box. Will return the center."""
    center_row = np.mean(box[:, 0])
    center_col = np.mean(box[:, 1])
    return center_row, center_col


def compute_sub_domain_position(sd_tracker_row: List, sd_tracker_col: List, sd_box: np.ndarray) -> List:
    """Given sub-domain displacements and sub-domain box. Will return the box center displacement at every step."""
    sd_row = []
    sd_col = []
    center_row, center_col = get_box_center(sd_box)
    num_frames = sd_tracker_row.shape[1]
    for kk in range(0, num_frames):
        row_pos = center_row + np.mean(sd_tracker_row[:, kk] - sd_tracker_row[:, 0])
        col_pos = center_col + np.mean(sd_tracker_col[:, kk] - sd_tracker_col[:, 0])
        sd_row.append(row_pos)
        sd_col.append(col_pos)
    return np.asarray(sd_row), np.asarray(sd_col)

def compute_region_center_displacement(
    sd_tracker_row: np.ndarray,
    sd_tracker_col: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute average region center displacement over time based on marker movement.

    Args:
        sd_tracker_row: (n_markers, n_frames) array of row positions over time.
        sd_tracker_col: (n_markers, n_frames) array of col positions over time.

    Returns:
        row_positions: np.ndarray of region row center at each time step
        col_positions: np.ndarray of region col center at each time step
    """

    if sd_tracker_row.ndim != 2 or sd_tracker_col.ndim != 2:
        raise ValueError("Inputs must be 2D arrays (n_markers, n_frames)")

    num_frames = sd_tracker_row.shape[1]

    # Center at frame 0
    center_row_0 = np.mean(sd_tracker_row[:, 0])
    center_col_0 = np.mean(sd_tracker_col[:, 0])

    row_positions = []
    col_positions = []

    for kk in range(num_frames):
        delta_row = np.mean(sd_tracker_row[:, kk] - sd_tracker_row[:, 0])
        delta_col = np.mean(sd_tracker_col[:, kk] - sd_tracker_col[:, 0])
        row_positions.append(center_row_0 + delta_row)
        col_positions.append(center_col_0 + delta_col)

    return np.array(row_positions), np.array(col_positions)

from scipy.spatial import cKDTree

def isolate_sub_domain_markers_regions(
    tracker_row_all: List[np.ndarray],
    tracker_col_all: List[np.ndarray],
    regions: np.ndarray,
    region_index: int
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Assign markers to a region by finding the closest pixel in `regions` array.

    Args:
        tracker_row_all: List of np.ndarray (N x ...), marker row coords (floats).
        tracker_col_all: List of np.ndarray (N x ...), marker col coords (floats).
        regions: 2D np.ndarray with integer region labels.
        region_index: integer region to isolate.

    Returns:
        sd_tracker_row_all, sd_tracker_col_all: Lists of arrays containing markers in the region.
    """

    # Precompute all pixel coordinates with a region label == region_index
    region_mask = (regions == region_index)
    region_coords = np.array(np.nonzero(region_mask)).T  # shape (num_pixels, 2)

    if len(region_coords) == 0:
        # No pixels for this region
        return [np.empty((0, tracker_row_all[0].shape[1])) for _ in tracker_row_all], \
               [np.empty((0, tracker_col_all[0].shape[1])) for _ in tracker_col_all]

    # Build a KD-tree for fast nearest neighbor search on the region pixels
    kd_tree = cKDTree(region_coords)

    sd_tracker_row_all = []
    sd_tracker_col_all = []

    for kk in range(len(tracker_row_all)):
        tracker_row = tracker_row_all[kk]
        tracker_col = tracker_col_all[kk]

        sd_tracker_row = []
        sd_tracker_col = []

        for jj in range(tracker_row.shape[0]):
            # Marker position (float)
            rr = tracker_row[jj, 0]
            cc = tracker_col[jj, 0]

            # Query nearest pixel in the target region
            dist, idx = kd_tree.query([rr, cc], k=1)
            nearest_pixel = region_coords[idx]

            # If the closest pixel is in the region, append the marker
            if dist < 5:  # threshold distance; adjust as needed
                sd_tracker_row.append(tracker_row[jj, :])
                sd_tracker_col.append(tracker_col[jj, :])

        sd_tracker_row_all.append(np.asarray(sd_tracker_row))
        sd_tracker_col_all.append(np.asarray(sd_tracker_col))

    return sd_tracker_row_all, sd_tracker_col_all

def plot_tracker_points_colored_by_region(tracker_row_all, tracker_col_all, regions):
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.cm as cm

    all_rows = []
    all_cols = []
    
    # Loop over all frames (or sets)
    for rows, cols in zip(tracker_row_all, tracker_col_all):
        # rows, cols can be (n_markers, n_frames) or (n_markers,) arrays
        # Flatten the arrays so we get all points as 1D arrays
        all_rows.append(rows.flatten())
        all_cols.append(cols.flatten())

    # Concatenate all points across all frames
    all_rows = np.concatenate(all_rows)
    all_cols = np.concatenate(all_cols)

    # Round or convert coordinates to int to index regions
    rr = np.round(all_rows).astype(int)
    cc = np.round(all_cols).astype(int)

    # Clip to valid indices to avoid indexing errors
    rr = np.clip(rr, 0, regions.shape[0]-1)
    cc = np.clip(cc, 0, regions.shape[1]-1)

    # Get region values at those points
    region_values = regions[rr, cc]

    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(all_cols, all_rows, c=region_values, cmap='tab20', s=10, alpha=0.8)
    plt.gca().invert_yaxis()  # Optional: depends on image coords

    plt.colorbar(scatter, label='Region Index')
    plt.title("Tracker points colored by region")
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.show()

def compute_sub_domain_position_strain_all(reg, tracker_row_all: List, tracker_col_all: List, sd_box_list: List) -> List:
    """Given all tracker rows, columns, and sub-domain box domains. Will return the markers within each sub-domain box."""
    num_beats = len(tracker_col_all)
    sub_domain_F_all = []
    sub_domain_row_all = []
    sub_domain_col_all = []
    unique_regions = np.unique(reg)
    unique_regions = unique_regions[unique_regions != 0]
    # for sd_box in sd_box_list:
    for i in unique_regions:
        # sd_tracker_row_all, sd_tracker_col_all = isolate_sub_domain_markers(tracker_row_all, tracker_col_all, sd_box)
        sd_tracker_row_all, sd_tracker_col_all = isolate_sub_domain_markers_regions(tracker_row_all, tracker_col_all, reg, i)
        F_list = []
        sd_row = []
        sd_col = []
        for kk in range(0, num_beats):
            sd_tracker_row = sd_tracker_row_all[kk]
            sd_tracker_col = sd_tracker_col_all[kk]
            if sd_tracker_row.shape[0] == 0:
                F_list.append(np.full((68, 4), np.nan))  # or continue
            else:
                F = compute_sub_domain_strain(sd_tracker_row, sd_tracker_col)
                F_list.append(F)
            # sd_row_kk, sd_col_kk = compute_sub_domain_position(sd_tracker_row, sd_tracker_col, sd_box)
            if sd_tracker_row.ndim != 2 or sd_tracker_col.ndim != 2 or sd_tracker_row.shape[0] == 0:
                sd_row_kk = np.full((68,), np.nan)
                sd_col_kk = np.full((68,), np.nan)
            else:
                sd_row_kk, sd_col_kk = compute_region_center_displacement(sd_tracker_row, sd_tracker_col)
            sd_row.append(sd_row_kk)
            sd_col.append(sd_col_kk)
        sub_domain_F_all.append(F_list)
        sub_domain_row_all.append(sd_row)
        sub_domain_col_all.append(sd_col)
    # plot_tracker_points_colored_by_region(tracker_row_all, tracker_col_all, reg)
    return sub_domain_F_all, sub_domain_row_all, sub_domain_col_all

# def compute_sub_domain_position_strain_all(tracker_row_all: List, tracker_col_all: List, sd_box_list: List) -> List:
#     """Given all tracker rows, columns, and sub-domain box domains. Will return the markers within each sub-domain box."""
#     num_beats = len(tracker_col_all)
#     sub_domain_F_all = []
#     sub_domain_row_all = []
#     sub_domain_col_all = []
#     for sd_box in sd_box_list:
#         sd_tracker_row_all, sd_tracker_col_all = isolate_sub_domain_markers(tracker_row_all, tracker_col_all, sd_box)
#         F_list = []
#         sd_row = []
#         sd_col = []
#         for kk in range(0, num_beats):
#             sd_tracker_row = sd_tracker_row_all[kk]
#             sd_tracker_col = sd_tracker_col_all[kk]
#             sd_F_array = compute_sub_domain_strain(sd_tracker_row, sd_tracker_col)
#             F_list.append(sd_F_array)
#             sd_row_kk, sd_col_kk = compute_sub_domain_position(sd_tracker_row, sd_tracker_col, sd_box)
#             sd_row.append(sd_row_kk)
#             sd_col.append(sd_col_kk)
#         sub_domain_F_all.append(F_list)
#         sub_domain_row_all.append(sd_row)
#         sub_domain_col_all.append(sd_col)
#     return sub_domain_F_all, sub_domain_row_all, sub_domain_col_all


def format_F_for_save(sub_domain_F_all: List) -> List:
    """Given sub_domain_F_all. Will return in a format where F_rr, F_rc, F_cr, and F_cc are formatted for saving."""
    """Subscript r refers to row and c to column: For example F_rr refers to F in the row-row direction."""
    sub_domain_F_rr_all = []
    sub_domain_F_rc_all = []
    sub_domain_F_cr_all = []
    sub_domain_F_cc_all = []
    num_beats = len(sub_domain_F_all[0])
    num_subdomains = len(sub_domain_F_all)
    for kk in range(0, num_beats):
        num_frames = sub_domain_F_all[0][kk].shape[0]
        sub_domain_F_rr = np.zeros((num_subdomains, num_frames))
        sub_domain_F_rc = np.zeros((num_subdomains, num_frames))
        sub_domain_F_cr = np.zeros((num_subdomains, num_frames))
        sub_domain_F_cc = np.zeros((num_subdomains, num_frames))
        for jj in range(0, num_subdomains):
            sub_domain_F_rr[jj, :] = sub_domain_F_all[jj][kk][:, 0]
            sub_domain_F_rc[jj, :] = sub_domain_F_all[jj][kk][:, 1]
            sub_domain_F_cr[jj, :] = sub_domain_F_all[jj][kk][:, 2]
            sub_domain_F_cc[jj, :] = sub_domain_F_all[jj][kk][:, 3]
        sub_domain_F_rr_all.append(sub_domain_F_rr)
        sub_domain_F_rc_all.append(sub_domain_F_rc)
        sub_domain_F_cr_all.append(sub_domain_F_cr)
        sub_domain_F_cc_all.append(sub_domain_F_cc)
    return sub_domain_F_rr_all, sub_domain_F_rc_all, sub_domain_F_cr_all, sub_domain_F_cc_all


def format_sd_row_col_for_save(sub_domain_row_all: List, sub_domain_col_all: List) -> List:
    """Given sub_domain_row_all and sub_domain_col_all. Will return in a format where sub_domain_row and sub_domain_col are formatted for saving."""
    sub_domain_row_all_new = []
    sub_domain_col_all_new = []
    num_beats = len(sub_domain_row_all[0])
    num_subdomains = len(sub_domain_row_all)
    for kk in range(0, num_beats):
        num_frames = sub_domain_row_all[0][kk].shape[0]
        sub_domain_row = np.zeros((num_subdomains, num_frames))
        sub_domain_col = np.zeros((num_subdomains, num_frames))
        for jj in range(0, num_subdomains):
            sub_domain_row[jj, :] = sub_domain_row_all[jj][kk]
            sub_domain_col[jj, :] = sub_domain_col_all[jj][kk]
        sub_domain_row_all_new.append(sub_domain_row)
        sub_domain_col_all_new.append(sub_domain_col)
    return sub_domain_row_all_new, sub_domain_col_all_new


def save_sub_domain_strain(folder_path: Path, sub_domain_F_all: List, sub_domain_row_all: List, sub_domain_col_all: List, strain_sub_domain_info: np.ndarray, *, fname: str = "") -> List:
    """Given results of sub domain strain computation. Will save the strain results."""
    new_path = ia.create_folder(folder_path, "results")
    num_beats = len(sub_domain_row_all[0])
    sub_domain_F_rr_all, sub_domain_F_rc_all, sub_domain_F_cr_all, sub_domain_F_cc_all = format_F_for_save(sub_domain_F_all)
    sub_domain_row_all, sub_domain_col_all = format_sd_row_col_for_save(sub_domain_row_all, sub_domain_col_all)
    saved_paths = []
    for kk in range(0, num_beats):
        # save the sub domain positions
        file_path = new_path.joinpath("strain_" + fname + "_beat%i_row.txt" % (kk)).resolve()
        saved_paths.append(file_path)
        np.savetxt(str(file_path), sub_domain_row_all[kk])
        file_path = new_path.joinpath("strain_" + fname + "_beat%i_col.txt" % (kk)).resolve()
        saved_paths.append(file_path)
        np.savetxt(str(file_path), sub_domain_col_all[kk])
        # save the components of the sub domain deformation gradient
        file_path = new_path.joinpath("strain_" + fname + "_beat%i_Frr.txt" % (kk)).resolve()
        saved_paths.append(file_path)
        np.savetxt(str(file_path), sub_domain_F_rr_all[kk])
        file_path = new_path.joinpath("strain_" + fname + "_beat%i_Frc.txt" % (kk)).resolve()
        saved_paths.append(file_path)
        np.savetxt(str(file_path), sub_domain_F_rc_all[kk])
        file_path = new_path.joinpath("strain_" + fname + "_beat%i_Fcr.txt" % (kk)).resolve()
        saved_paths.append(file_path)
        np.savetxt(str(file_path), sub_domain_F_cr_all[kk])
        file_path = new_path.joinpath("strain_" + fname + "_beat%i_Fcc.txt" % (kk)).resolve()
        saved_paths.append(file_path)
        np.savetxt(str(file_path), sub_domain_F_cc_all[kk])
    # save the information about the sub domains (specifically: num_row_tile, num_col_tile, tile_pix_dim)
    file_path = new_path.joinpath("strain_" + fname + "_sub_domain_info.txt").resolve()
    np.savetxt(str(file_path), strain_sub_domain_info)
    saved_paths.append(file_path)
    return saved_paths


def load_sub_domain_strain(folder_path: Path, fname="") -> List:
    """Given folder path. Will load strain results. If there are none, will return an error."""
    res_folder_path = folder_path.joinpath("results").resolve()
    if res_folder_path.exists() is False:
        raise FileNotFoundError("tracking results are not present -- therefore strain results must not be present either")
    file_list = glob.glob(str(res_folder_path) + "/*strain*")
    if len(file_list) == 0:
        raise FileNotFoundError("strain results are not present")
    num_files = len(glob.glob(str(res_folder_path) + "/*strain*beat*.txt"))
    num_beats = int((num_files) / 6)
    sub_domain_F_rr_all = []
    sub_domain_F_rc_all = []
    sub_domain_F_cr_all = []
    sub_domain_F_cc_all = []
    sub_domain_row_all = []
    sub_domain_col_all = []
    for kk in range(0, num_beats):
        sub_domain_F_rr_all.append(np.loadtxt(str(res_folder_path) + "/strain_" + fname + "_beat%i_Frr.txt" % (kk)))
        sub_domain_F_rc_all.append(np.loadtxt(str(res_folder_path) + "/strain_" + fname + "_beat%i_Frc.txt" % (kk)))
        sub_domain_F_cr_all.append(np.loadtxt(str(res_folder_path) + "/strain_" + fname + "_beat%i_Fcr.txt" % (kk)))
        sub_domain_F_cc_all.append(np.loadtxt(str(res_folder_path) + "/strain_" + fname + "_beat%i_Fcc.txt" % (kk)))
        sub_domain_row_all.append(np.loadtxt(str(res_folder_path) + "/strain_" + fname + "_beat%i_row.txt" % (kk)))
        sub_domain_col_all.append(np.loadtxt(str(res_folder_path) + "/strain_" + fname + "_beat%i_col.txt" % (kk)))
    strain_info = np.loadtxt(str(res_folder_path) + "/strain_" + fname + "_sub_domain_info.txt")
    info = np.loadtxt(str(res_folder_path) + "/info.txt")
    info_reshape = np.reshape(info, (-1, 3))
    return sub_domain_F_rr_all, sub_domain_F_rc_all, sub_domain_F_cr_all, sub_domain_F_cc_all, sub_domain_row_all, sub_domain_col_all, info_reshape, strain_info


def F_to_E(F_rr: float, F_rc: float, F_cr: float, F_cc: float) -> float:
    """Given F, will compute E_cc (E_column_column) for visualization."""
    F_arr = np.asarray([[F_rr, F_rc], [F_cr, F_cc]])
    C = np.dot(F_arr.T, F_arr)
    E = 0.5 * (C - np.eye(2))
    return E[1, 1], E[1,0], E[0,0]


def F_to_E_all(sub_domain_F_rr_all: List, sub_domain_F_rc_all: List, sub_domain_F_cr_all: List, sub_domain_F_cc_all: List) -> List:
    sub_domain_Ecc_all = []
    sub_domain_Ecr_all = []
    sub_domain_Err_all = []
    num_beats = len(sub_domain_F_rr_all)
    num_sub_domains = sub_domain_F_rr_all[0].shape[0]
    for kk in range(0, num_beats):
        num_frames = sub_domain_F_rr_all[kk].shape[1]
        Ecc_arr = np.zeros((num_sub_domains, num_frames))
        Ecr_arr = np.zeros((num_sub_domains, num_frames))
        Err_arr = np.zeros((num_sub_domains, num_frames))
        for jj in range(0, num_sub_domains):
            for ii in range(0, num_frames):
                F_rr = sub_domain_F_rr_all[kk][jj, ii]
                F_rc = sub_domain_F_rc_all[kk][jj, ii]
                F_cr = sub_domain_F_cr_all[kk][jj, ii]
                F_cc = sub_domain_F_cc_all[kk][jj, ii]
                Ecc_arr[jj, ii], Ecr_arr[jj, ii], Err_arr[jj, ii]= F_to_E(F_rr, F_rc, F_cr, F_cc)
        sub_domain_Ecc_all.append(Ecc_arr)
        sub_domain_Ecr_all.append(Ecr_arr)
        sub_domain_Err_all.append(Err_arr)
    return sub_domain_Ecc_all, sub_domain_Ecr_all, sub_domain_Err_all


def get_text_str(row: int, col: int) -> str:
    """Given the row and the column of a location. Will return the row and column position string."""
    test_str = "%s%i" % (chr(row + 65), col + 1)
    return test_str



def get_text_str_from_index(idx: int, num_cols: int = 10) -> str:
    row = idx // num_cols
    col = idx % num_cols
    return get_text_str(row, col)

def png_sub_domains_numbered(
    folder_path: Path,
    example_tiff: np.ndarray,
    regions: np.ndarray,
    fname: str = "strain_",
    col_map=plt.cm.rainbow
) -> List:
    import matplotlib.patheffects as pe
    from skimage import measure

    vis_folder_path = ia.create_folder(folder_path, "visualizations")
    pngs_folder_path = ia.create_folder(vis_folder_path, "strain_pngs")
    img_path = pngs_folder_path.joinpath(fname + "sub_domain_key.pdf").resolve()
    plt.figure()
    plt.imshow(example_tiff, cmap=plt.cm.gray)
    
    region_labels = np.unique(regions)
    region_labels = region_labels[region_labels != 0]  # exclude background if labeled 0
    
    for idx, label in enumerate(region_labels):
        mask = (regions == label)
        contours = measure.find_contours(mask.astype(float), 0.5)
        if len(contours) == 0:
            continue
        contour = contours[0]  # might be multiple contours, take first or all
        
        # contour is (N, 2) array: rows, cols
        plt.plot(contour[:, 1], contour[:, 0], color=col_map(idx / len(region_labels)), linewidth=1)
        
        # Compute centroid for label placement
        coords = np.column_stack(np.where(mask))
        center_row, center_col = np.mean(coords, axis=0)
        
        text_str = get_text_str_from_index(idx)  # your function or adapt
        plt.text(center_col, center_row, text_str, color=col_map(idx / len(region_labels)),
                 fontsize=12, horizontalalignment="center", verticalalignment="center",
                 path_effects=[pe.Stroke(linewidth=1, foreground='k'), pe.Normal()])
    
    plt.axis("off")
    plt.savefig(str(img_path), format='pdf')
    plt.close()
    return img_path


# def png_sub_domains_numbered(
#     folder_path: Path,
#     example_tiff: np.ndarray,
#     sub_domain_row: List,
#     sub_domain_col: List,
#     sub_domain_side: Union[float, int],
#     num_sd_row: int,
#     num_sd_col: int,
#     fname: str = "strain_",
#     col_map: object = plt.cm.rainbow
# ) -> List:
#     """Given information to visualize the sub-domains. Will plot the subdomains and label them.
#     Rows are labeled A, B, C, etc. -- columns are labeled 1, 2, 3, etc. """
#     vis_folder_path = ia.create_folder(folder_path, "visualizations")
#     pngs_folder_path = ia.create_folder(vis_folder_path, "strain_pngs")
#     img_path = pngs_folder_path.joinpath(fname + "sub_domain_key.pdf").resolve()
#     plt.figure()
#     plt.imshow(example_tiff, cmap=plt.cm.gray)
#     sds = sub_domain_side / 2.0
#     for cc in range(0, num_sd_col):
#         for rr in range(0, num_sd_row):
#             idx = rr * num_sd_col + cc
#             center_row = sub_domain_row[idx, 0]
#             center_col = sub_domain_col[idx, 0]
#             text_str = get_text_str(rr, cc)
#             plt.text(center_col, center_row, text_str, color=col_map(idx / (num_sd_col * num_sd_row)), fontsize= int(sub_domain_side/3) ,horizontalalignment="center", verticalalignment="center",path_effects=[pe.Stroke(linewidth=1, foreground='k'), pe.Normal()])
#             corners_rr = [center_row - sds, center_row - sds, center_row + sds, center_row + sds, center_row - sds]
#             corners_cc = [center_col - sds, center_col + sds, center_col + sds, center_col - sds, center_col - sds]
#             plt.plot(corners_cc, corners_rr, "k-", linewidth=1)
#     plt.axis("off")
#     plt.savefig(str(img_path),format='pdf')
#     plt.close()
#     return img_path


def png_sub_domain_strain_timeseries_all(
    folder_path: Path,
    sub_domain_strain_all: List,
    num_sd_row: int,
    num_sd_col: int,
    output: str,
    col_map: object = plt.cm.rainbow,
    *,
    fname: str = "strain_timeseries_Ecc",
    xlabel: str = "frame",
    ylabel: str = "strain Ecc"
) -> List:
    """Given strain timeseries. Will plot all timeseries on the same axis."""
    vis_folder_path = ia.create_folder(folder_path, "visualizations")
    main_pngs_folder_path = ia.create_folder(vis_folder_path, "strain_pngs")
    
    if output == "Ecc":
        pngs_folder_path = ia.create_folder(main_pngs_folder_path, "Ecc")
    elif output == "Ecr":
        pngs_folder_path = ia.create_folder(main_pngs_folder_path, "Ecr")  
    elif output == "Err":
        pngs_folder_path = ia.create_folder(main_pngs_folder_path, "Err")           
    num_beats = len(sub_domain_strain_all)
    path_list = []
    for kk in range(0, num_beats):
        plt.figure()
        ax = plt.subplot(111)
        sub_domain_strain = sub_domain_strain_all[kk]
        for cc in range(0, num_sd_col):
            for rr in range(0, num_sd_row):
                lab = get_text_str(rr, cc)
                idx = rr * num_sd_col + cc
                ax.plot(sub_domain_strain[idx, :], label=lab, color=col_map(idx/(num_sd_col * num_sd_row)))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title("beat %i" % (kk))
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.25, box.width, box.height * 0.75])
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=num_sd_col)
        img_path = pngs_folder_path.joinpath(fname + "_beat%i.pdf" % (kk)).resolve()
        plt.savefig(str(img_path), format='pdf')
        plt.close()
        path_list.append(img_path)
    return path_list


def compute_min_max_strain (
    sub_domain_E_all,
    info: np.ndarray
) -> float: 
    """Given tracking results. Will find the minimum and maximum displacement over all beats."""
    num_beats = info.shape[0]
    min_E_all,max_E_all = [],[]
    for beat in range(0, num_beats):
        E = sub_domain_E_all[beat]
        min_E_all.append(np.min(E))
        max_E_all.append(np.max(E))
    min_E_c = np.min(min_E_all)
    max_E_c = np.max(max_E_all)
    
    return min_E_c, max_E_c
   

def pngs_sub_domain_strain(
    folder_path: Path,
    tiff_list: List,
    sub_domain_row_all: List,
    sub_domain_col_all: List,
    sub_domain_E_all: List,
    sub_domain_side: Union[float, int],
    info: np.ndarray,
    output: str,
    col_min: Union[float, int] = -0.025,
    col_max: Union[float, int] = 0.025,
    col_map: object = plt.cm.RdBu,
    fname: str = "strain",
    save_eps: bool = False
) -> List:
    """Given sub domain strain results. Will create pngs."""
    vis_folder_path = ia.create_folder(folder_path, "visualizations")
    main_pngs_folder_path = ia.create_folder(vis_folder_path, "strain_pngs")
    
    if output == "Ecc":
        pngs_folder_path = ia.create_folder(main_pngs_folder_path, "Ecc")
        E_label = "Ecc"
    elif output == "Ecr":
        pngs_folder_path = ia.create_folder(main_pngs_folder_path, "Ecr")  
        E_label = "Ecr"
    elif output == "Err":
        pngs_folder_path = ia.create_folder(main_pngs_folder_path, "Err") 
        E_label = "Err"
        
    path_list = []
    num_beats = info.shape[0]
    for beat in range(0, num_beats):
        tracker_row = sub_domain_row_all[beat]
        tracker_col = sub_domain_col_all[beat]
        E = sub_domain_E_all[beat]
        start_idx = int(info[beat, 1])
        end_idx = int(info[beat, 2])
        for kk in range(start_idx, end_idx):
            plt.figure()
            plt.imshow(tiff_list[kk], cmap=plt.cm.gray)
            jj = kk - start_idx
            plt.scatter(tracker_col[:, jj], tracker_row[:, jj], c=E[:, jj], s=50, cmap=col_map, vmin=col_min, vmax=col_max)
            # illustrate sub-domain borders
            sds = sub_domain_side / 2.0
            for ii in range(0, tracker_col.shape[0]):
                corners_rr = [tracker_row[ii, jj] - sds, tracker_row[ii, jj] - sds, tracker_row[ii, jj] + sds, tracker_row[ii, jj] + sds, tracker_row[ii, jj] - sds]
                corners_cc = [tracker_col[ii, jj] - sds, tracker_col[ii, jj] + sds, tracker_col[ii, jj] + sds, tracker_col[ii, jj] - sds, tracker_col[ii, jj] - sds]
                plt.plot(corners_cc, corners_rr, "k-", linewidth=0.1)
                
            plt.title("frame %i, beat %i, %s" % (kk, beat, E_label))
            cbar = plt.colorbar()
            cbar.ax.get_yaxis().labelpad = 15
            cbar.set_label("%s strain (image column axis)"%(E_label), rotation=270)
            plt.axis("off")
            path = pngs_folder_path.joinpath("%04d_" % (kk) + fname + ".png").resolve()
            plt.savefig(str(path))
            if save_eps:
                plt.savefig(str(path)[0:-4]+'.eps',format='eps')
            plt.close()
            path_list.append(path)
    return path_list


def create_gif(folder_path: Path, png_path_list: List, fname="sub_domain_strain") -> Path:
    """Given the pngs path list. Will create a gif."""
    img_list = []
    img = plt.imread(png_path_list[0])
    img_r, img_c,_ = img.shape
    fig, ax = plt.subplots(figsize=(img_c/100,img_r/100))
    plt.axis('off') 
    plt.tight_layout(pad=0.08, h_pad=None, w_pad=None, rect=None)
    for pa in png_path_list:
        img = ax.imshow(plt.imread(pa),animated=True)
        img_list.append([img])
    fn_gif = fname + ".gif"
    gif_path = folder_path.joinpath("visualizations").resolve().joinpath(fn_gif).resolve()
    ani = animation.ArtistAnimation(fig, img_list,interval=100)
    ani.save(gif_path,dpi=100)
    return gif_path

from skimage.morphology import erosion, rectangle
import matplotlib.pyplot as plt

def erode_vertical_waist_in_horizontal_band(mask, waist_fraction=0.6, erosion_radius=10):
    """
    Erodes the mask vertically (top and bottom) but only in the horizontal waist region.
    
    Params:
    - mask: 2D binary mask (np.ndarray)
    - waist_fraction: fraction of the mask width to consider as waist (centered)
    - erosion_radius: radius of vertical erosion
    
    Returns:
    - eroded_mask: mask with vertically eroded waist
    """
    height, width = mask.shape

    # Define waist horizontal range (centered)
    waist_width = int(width * waist_fraction)
    left = (width - waist_width) // 2
    right = left + waist_width
    waist_slice = slice(left, right)

    # Extract just the waist region horizontally
    waist_region = np.zeros_like(mask)
    waist_region[:, waist_slice] = mask[:, waist_slice]

    # Create a vertical structuring element for erosion (taller than wider)
    selem = rectangle(erosion_radius * 2 + 1, 1)  # tall vertical line

    # Erode only the waist region vertically
    eroded_waist = erosion(waist_region, selem)

    # Compose final mask:
    eroded_mask = mask.copy()
    eroded_mask[:, waist_slice] = eroded_waist[:, waist_slice]

    return eroded_mask

def custom_region_segmentation(mask):
    import numpy as np
    import matplotlib.pyplot as plt

    def get_utc_images(bk):
        i = np.arange(bk.shape[0])
        j = np.arange(bk.shape[1])
        I, J = np.meshgrid(j, i)

        axis_dir = J / np.max(J)
        trans_dir = np.zeros_like(axis_dir, dtype=float)  # make sure it's float for fractional values

        for row_idx in range(J.shape[0]):
            positive_indices = np.where(bk[row_idx, :] > 0)[0]
            if positive_indices.size == 0:
                # No positive values in this row, assign 0 or some default
                trans_dir[row_idx] = 0
                continue

            lims = positive_indices[[0, -1]]
            size = lims[1] - lims[0]

            if size == 0:
                # All positive values are at a single point
                trans_dir[row_idx] = 0  # or 1, depending on logic
            else:
                trans_dir[row_idx] = (I[row_idx] - lims[0]) / size

        # Clamp values between 0 and 1
        trans_dir = np.clip(trans_dir, 0, 1)

        return axis_dir, trans_dir

    nx, ny = 5, 10
    nregions = nx*ny

    # mask = ia.remove_pillar_region(mask, 0.6, clip_columns=False, clip_rows=True)
    # mask = ia.remove_pillar_region(mask, 0.15, clip_columns=True, clip_rows=False)

    axis_dir, trans_dir = get_utc_images(mask)

    # Mask valid pixels
    valid_pixels = mask > 0

    # Extract axis_dir and trans_dir values only inside the mask
    axis_valid = axis_dir[valid_pixels]
    trans_valid = trans_dir[valid_pixels]

    # Compute quantiles to split data into nx and ny bins with equal pixels
    xlims = np.quantile(axis_valid, np.linspace(0, 1, nx + 1))
    ylims = np.quantile(trans_valid, np.linspace(0, 1, ny + 1))

    regions = np.zeros_like(mask, dtype=int)

    for i in range(nx):
        for j in range(ny):
            region_mask = (
                (axis_dir >= xlims[i]) & (axis_dir < xlims[i + 1]) &
                (trans_dir >= ylims[j]) & (trans_dir < ylims[j + 1]) &
                valid_pixels
            )
            regions[region_mask] = i + j * nx + 1

    # Optional: assign zero to pixels outside mask explicitly
    regions[~valid_pixels] = 0

    # plt.figure()
    # plt.imshow(regions, cmap='jet', vmin=0, vmax=nregions)
    # plt.axis('off')
    # plt.show()

    return regions

def extract_tile_info(regions: np.ndarray):
    import scipy.ndimage as ndi

    unique_regions = np.unique(regions)
    unique_regions = unique_regions[unique_regions != 0]

    objects = ndi.find_objects(regions)

    tile_heights = []
    tile_widths = []

    for sl in objects:
        if sl is None:
            continue
        tile_heights.append(sl[0].stop - sl[0].start)
        tile_widths.append(sl[1].stop - sl[1].start)

    tile_dim_pix = (int(np.median(tile_heights)), int(np.median(tile_widths)))

    centroids = ndi.center_of_mass(regions > 0, labels=regions, index=unique_regions)
    rows = np.array([c[0] for c in centroids])
    cols = np.array([c[1] for c in centroids])

    num_tile_row = len(np.unique(np.round(rows).astype(int)))
    num_tile_col = len(np.unique(np.round(cols).astype(int)))

    return num_tile_row, num_tile_col, tile_dim_pix


def run_sub_domain_strain_analysis(
    folder_path: Path,
    pillar_clip_fraction: float = 0.5,
    shrink_row: float = 0.1,
    shrink_col: float = 0.1,
    tile_dim_pix: int = 40,
    num_tile_row: int = 5,
    num_tile_col: int = 3,
    tile_style: int = 1,
    is_rotated: bool = True,
    clip_columns: bool = True,
    clip_rows: bool = False,
    manual_sub: bool = False,
    sub_extents: List = None,
    *,
    save_fname: str = ""
) -> List:
    """Given a folder path. Will perform strain analysis and save results as text files.
    Note that this function assumes that we have already run the tracking portion of the code."""
    # read images and mask file
    mask_file_path = folder_path.joinpath("masks").resolve().joinpath("tissue_mask.txt").resolve()
    mask = ia.read_txt_as_mask(mask_file_path)
    # load tracking results
    tracker_row_all, tracker_col_all, _, _ = ia.load_tracking_results(folder_path=folder_path)
    
    if is_rotated:
        # rotate tracking results
        (center_row, center_col, rot_mat, ang, vec) = ia.get_rotation_info(mask=mask)
        rot_tracker_row_all, rot_tracker_col_all = ia.rotate_pts_all(tracker_row_all, tracker_col_all, rot_mat, center_row, center_col)
        if abs(ang) < 0.96*np.pi:
            square = ia.check_square_image(mask)
            if square == False:
                # pad mask
                padded_mask, translate_r, translate_c = ia.pad_img_to_square(mask)
                # translate center of rotation 
                trans_center_row, trans_center_col = ia.translate_points(center_row,center_col,translate_r,translate_c)
                # rotate mask
                rot_mask  = ia.rot_image(padded_mask, trans_center_row, trans_center_col, ang)
                # translate rotated results 
                rot_tracker_row_all_pad, rot_tracker_col_all_pad = ia.translate_pts_all(rot_tracker_row_all,rot_tracker_col_all,translate_r,translate_c)
            else:
                rot_mask  = ia.rot_image(mask, center_row, center_col, ang)
                rot_tracker_row_all_pad, rot_tracker_col_all_pad = rot_tracker_row_all, rot_tracker_col_all        
        else:
            rot_mask = mask
            rot_tracker_row_all_pad, rot_tracker_col_all_pad = tracker_row_all, tracker_col_all

        rot_mask = erode_vertical_waist_in_horizontal_band(rot_mask, erosion_radius=7)
        seg_regions = custom_region_segmentation(rot_mask)
        # np.save('./rot_mask.npy', rot_mask)
        
        # create sub-domains
        sub_domain_box_list, tile_dim_pix, num_tile_row, num_tile_col = create_sub_domains(rot_mask, pillar_clip_fraction=pillar_clip_fraction, shrink_row=shrink_row, shrink_col=shrink_col, tile_dim_pix=tile_dim_pix, num_tile_row=num_tile_row, num_tile_col=num_tile_col, tile_style=tile_style,clip_columns=clip_columns,clip_rows=clip_rows,manual_sub= manual_sub,sub_extents=sub_extents) 
        # compute strain in each sub-domain
        sub_domain_F_all, sub_domain_row_all, sub_domain_col_all = compute_sub_domain_position_strain_all(seg_regions, rot_tracker_row_all_pad, rot_tracker_col_all_pad, sub_domain_box_list) 
        # save the sub-domain strains
        num_tile_row, num_tile_col, tile_dim_pix = extract_tile_info(seg_regions)
        strain_sub_domain_info = np.asarray([[num_tile_row, num_tile_col], [tile_dim_pix[0], tile_dim_pix[1]], [center_row, center_col], [vec[0], vec[1]]])
    else:
        # create sub-domains
        sub_domain_box_list, tile_dim_pix, num_tile_row, num_tile_col = create_sub_domains(mask, pillar_clip_fraction=pillar_clip_fraction, shrink_row=shrink_row, shrink_col=shrink_col, tile_dim_pix=tile_dim_pix, num_tile_row=num_tile_row, num_tile_col=num_tile_col, tile_style=tile_style,clip_columns=clip_columns,clip_rows=clip_rows,manual_sub= manual_sub,sub_extents=sub_extents)
        # compute strain in each sub-domain
        sub_domain_F_all, sub_domain_row_all, sub_domain_col_all = compute_sub_domain_position_strain_all(tracker_row_all, tracker_col_all, sub_domain_box_list)
        # save the sub-domain strains
        strain_sub_domain_info = np.asarray([[num_tile_row, num_tile_col], [tile_dim_pix, tile_dim_pix]])
   
    saved_paths = save_sub_domain_strain(folder_path, sub_domain_F_all, sub_domain_row_all, sub_domain_col_all, strain_sub_domain_info, fname=save_fname)
    return saved_paths


def visualize_sub_domain_strain(
    folder_path: Path,
    automatic_color_constraint: bool = True,
    col_min: Union[float, int] = -0.025,
    col_max: Union[float, int] = 0.025,
    col_map: object = plt.cm.RdBu,
    fname: str = "strain",
    is_rotated: bool = True,
) -> List:
    """Given a folder path where strain analysis has already been run. Will save visualizations."""
    # read image files
    movie_folder_path = folder_path.joinpath("movie").resolve()
    name_list_path = ia.image_folder_to_path_list(movie_folder_path)
    tiff_list = ia.read_all_tiff(name_list_path)
    # load the strain results
    sub_domain_F_rr_all, sub_domain_F_rc_all, sub_domain_F_cr_all, sub_domain_F_cc_all, sub_domain_row_all, sub_domain_col_all, info, strain_info = load_sub_domain_strain(folder_path)
    
    num_sd_row = int(strain_info[0, 0])
    num_sd_col = int(strain_info[0, 1])
    sub_domain_side = strain_info[1, 0]
    
    if is_rotated:
        # rotate the background tiff to match the strain results
        center_row = strain_info[2, 0]
        center_col = strain_info[2, 1]
        vec = strain_info[3, :]
        # strain_sub_domain_info = np.asarray([[num_tile_row, num_tile_col], [tile_dim_pix, tile_dim_pix], [center_row, center_col], [vec[0], vec[1]]])
        (_, ang) = ia.rot_vec_to_rot_mat_and_angle(vec)
        if abs(ang) < 0.96*np.pi:
            square = ia.check_square_image(tiff_list[0])
            if square == False:
                # pad image list
                padded_tiff_list, translate_r, translate_c = ia.pad_all_imgs_to_square(tiff_list)
                # translate center of rotation
                trans_center_row, trans_center_col = ia.translate_points(center_row,center_col,translate_r,translate_c)
                # rotate all images in list
                tiff_list = ia.rotate_imgs_all(padded_tiff_list, ang, trans_center_row, trans_center_col)
            else:
                tiff_list = ia.rotate_imgs_all(tiff_list, ang, center_row, center_col)
        else:
            tiff_list = tiff_list

    # convert the strain results to Ecc for plotting
    sub_domain_Ecc_all, sub_domain_Ecr_all, sub_domain_Err_all = F_to_E_all(sub_domain_F_rr_all, sub_domain_F_rc_all, sub_domain_F_cr_all, sub_domain_F_cc_all)
    if automatic_color_constraint:
        # find limits of colormap
        clim_Ecc_min, clim_Ecc_max = compute_min_max_strain(sub_domain_Ecc_all,info)
        clim_Ecr_min, clim_Ecr_max = compute_min_max_strain(sub_domain_Ecr_all,info)
        clim_Err_min, clim_Err_max = compute_min_max_strain(sub_domain_Err_all,info)
    # create png visualizations
    png_path_list_Ecc = pngs_sub_domain_strain(folder_path, tiff_list, sub_domain_row_all, sub_domain_col_all, sub_domain_Ecc_all, sub_domain_side, info, "Ecc", clim_Ecc_min, clim_Ecc_max, col_map, fname, save_eps=False)
    png_path_list_Ecr = pngs_sub_domain_strain(folder_path, tiff_list, sub_domain_row_all, sub_domain_col_all, sub_domain_Ecr_all, sub_domain_side, info, "Ecr", clim_Ecr_min, clim_Ecr_max, col_map, fname, save_eps=False)
    png_path_list_Err = pngs_sub_domain_strain(folder_path, tiff_list, sub_domain_row_all, sub_domain_col_all, sub_domain_Err_all, sub_domain_side, info, "Err", clim_Err_min, clim_Err_max, col_map, fname, save_eps=False)    
    # create gif
    gif_path_Ecc = create_gif(folder_path, png_path_list_Ecc, fname="sub_domain_strain_Ecc")
    gif_path_Ecr = create_gif(folder_path, png_path_list_Ecr, fname="sub_domain_strain_Ecr")
    gif_path_Err = create_gif(folder_path, png_path_list_Err, fname="sub_domain_strain_Err")
    # create subdomain locations legend
    loc_legend_path = png_sub_domains_numbered(folder_path, tiff_list[0], sub_domain_row_all[0], sub_domain_col_all[0], sub_domain_side, num_sd_row, num_sd_col)
    # create subdomain strain timeseries plots
    timeseries_path_list_Ecc = png_sub_domain_strain_timeseries_all(folder_path, sub_domain_Ecc_all, num_sd_row, num_sd_col, output="Ecc", fname="strain_timeseries_Ecc", ylabel="strain Ecc")
    timeseries_path_list_Ecr = png_sub_domain_strain_timeseries_all(folder_path, sub_domain_Ecr_all, num_sd_row, num_sd_col, output="Ecr", fname="strain_timeseries_Ecr", ylabel="strain Ecr")
    timeseries_path_list_Err = png_sub_domain_strain_timeseries_all(folder_path, sub_domain_Err_all, num_sd_row, num_sd_col, output="Err", fname="strain_timeseries_Err", ylabel="strain Err")
    return png_path_list_Ecc, png_path_list_Ecr, png_path_list_Err, gif_path_Ecc, gif_path_Ecr, gif_path_Err, loc_legend_path, timeseries_path_list_Ecc, timeseries_path_list_Ecr, timeseries_path_list_Err

 


