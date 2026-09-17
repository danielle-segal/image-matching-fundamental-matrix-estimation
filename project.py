import os
import csv
import cv2
import torch
import numpy as np
from tqdm import tqdm

BASE_PATH = "/kaggle/input"
subproject_folder = "cv-22928-2025-a-project"
src = os.path.join(BASE_PATH, subproject_folder)

if os.path.exists(src):
    print(f"Successfully found project directory at: {src}")
else:
    print(f"ERROR: Could not find project directory at: {src}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

if torch.cuda.is_available():
    print(f'GPU Name: {torch.cuda.get_device_name(0)}')
    print(f'Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')

from romatch import roma_outdoor
roma_model = roma_outdoor(device=device)

from torch.utils.data import Dataset, DataLoader


class ImagePairDataset(Dataset):
    """Dataset of image-pair references loaded from the competition's test CSV.

    Each row of the CSV is resolved to the on-disk paths of the two images that
    make up that sample; images themselves are loaded later, inside
    `process_batch`, rather than in `__getitem__`.
    """

    def __init__(self, csv_file, project_dir):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            project_dir (string): Directory with all the images.
        """
        with open(csv_file) as f:
            reader = csv.reader(f, delimiter=',')
            next(reader)
            self.samples = list(reader)
        self.project_dir = project_dir

    def __len__(self):
        """Return the number of image-pair samples in the dataset.

        Returns:
            int: Number of rows loaded from the csv file.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """Resolve a sample index to its image-pair file paths.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: Dictionary with 'sample_id' (str), 'img1_path' (string), and
            'img2_path' (string) for the requested pair.
        """
        sample_id, batch_id, image_1_id, image_2_id = self.samples[idx]
        img1_path = os.path.join(self.project_dir, 'test_images', batch_id, f'{image_1_id}.jpg')
        img2_path = os.path.join(self.project_dir, 'test_images', batch_id, f'{image_2_id}.jpg')

        return {
            'sample_id': sample_id,
            'img1_path': img1_path,
            'img2_path': img2_path
        }


def flatten_matrix(M, num_digits=8):
    """Flatten a 3x3 matrix into the submission's space-separated string format.

    Args:
        M (np.ndarray): Matrix to serialize (row-major flatten order).
        num_digits (int, optional): Number of digits for scientific notation.
            Defaults to 8.

    Returns:
        str: Space-separated scientific-notation representation of `M`.
    """
    return ' '.join([f'{v:.{num_digits}e}' for v in M.flatten()])  # convert matrix to string format for submission


def process_batch(batch, model, device):
    """Estimate the fundamental matrix for every image pair in a batch.

    For each sample: runs RoMa to obtain dense correspondences, fits an initial
    fundamental matrix with RANSAC (USAC_MAGSAC), then refines it using the
    8-point algorithm on the inlier set. Any failure (missing/corrupt images,
    insufficient inliers, or a degenerate fit) falls back to the identity matrix
    so that a single bad sample never aborts the run.

    Args:
        batch (list[dict]): Samples from `ImagePairDataset`, each with
            'sample_id', 'img1_path', and 'img2_path'.
        model: Initialized RoMa model used for matching.
        device: Torch device to run matching on.

    Returns:
        dict[str, np.ndarray]: Mapping from sample_id to its estimated (or
        fallback) 3x3 fundamental matrix.
    """
    results = {}

    for sample in batch:  # Process a batch of image pairs using RoMa with guided matching refinement
        sample_id = sample['sample_id']
        img1_path = sample['img1_path']
        img2_path = sample['img2_path']

        try:
            # to check images exist
            if not os.path.exists(img1_path) or not os.path.exists(img2_path):
                results[sample_id] = np.eye(3)  # identity matrix
                continue

            # to get the image's dimensions
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)
            if img1 is None or img2 is None:
                results[sample_id] = np.eye(3)
                continue

            H_A, W_A = img1.shape[:2]
            H_B, W_B = img2.shape[:2]

            # match with RoMa
            warp, certainty = model.match(img1_path, img2_path, device=device)

            # to sample the matches
            matches, certainty = model.sample(warp, certainty)

            # convert the keypoints to pixel coordinates
            kptsA, kptsB = model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)

            # initial fundamental matrix using RANSAC
            F, mask = cv2.findFundamentalMat(
                kptsA.cpu().numpy(),
                kptsB.cpu().numpy(),
                ransacReprojThreshold=0.5,
                method=cv2.USAC_MAGSAC,
                confidence=0.999999,
                maxIters=100000
            )

            # second and better F using refinement
            if F is not None and F.shape == (3, 3) and mask is not None:
                # mask convertion to the right format, and find inliers
                mask = mask.ravel() == 1
                if np.sum(mask) >= 8:  # FM_8POINT constraint
                    inlier_pts1 = kptsA.cpu().numpy()[mask]
                    inlier_pts2 = kptsB.cpu().numpy()[mask]

                    # refining F using only inliers with the 8-point algorithm
                    F_refined, _ = cv2.findFundamentalMat(
                        inlier_pts1,
                        inlier_pts2,
                        method=cv2.FM_8POINT
                    )

                    if F_refined is not None and F_refined.shape == (3, 3):
                        F = F_refined

            if F is None or F.shape != (3, 3):
                results[sample_id] = np.eye(3)
            else:
                results[sample_id] = F

        except Exception as e:
            print(f"Error processing {sample_id}: {str(e)}")
            results[sample_id] = np.eye(3)

    return results

def main():
    """Run the full pipeline: load test samples, estimate fundamental matrices
    for every pair, and write the submission CSV.

    Builds the `ImagePairDataset` / `DataLoader` over `test.csv`, runs
    `process_batch` on each batch, and writes all results to
    `/kaggle/working/submission.csv`. Exits early (without writing a file) if
    `test.csv` cannot be found.
    """

    test_csv = os.path.join(src, 'test.csv')

    if not os.path.exists(test_csv):
        print(f"Error: test.csv not found at {test_csv}")
        return

    # create the dataset and the dataloader
    dataset = ImagePairDataset(test_csv, src)
    batch_size = 4
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=lambda x: x  # to keep the original dictionary's format
    )

    all_results = {}

    for batch in tqdm(dataloader, desc='Processing batches'):
        batch_results = process_batch(batch, roma_model, device)
        all_results.update(batch_results)

    submission_path = os.path.join('/kaggle/working', 'submission.csv')
    with open(submission_path, 'w') as f:
        f.write('sample_id,fundamental_matrix\n')
        for sample_id, F in all_results.items():
            f.write(f'{sample_id},{flatten_matrix(F)}\n')

    print(f"Submission saved to: {submission_path}")


if __name__ == "__main__":
    main()
    print("Processing complete! Submission file has been created.")