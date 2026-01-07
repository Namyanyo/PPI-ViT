
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from PIL import Image


class PointCloudProjector:
    def __init__(self, image_size=224, method='density_map'):
        """
        Initialize the projector

        Args:
            image_size: Output image size (image_size x image_size)
            method: Projection method ('density_map', 'scatter', 'voxel')
        """
        self.image_size = image_size
        self.method = method

    def project_to_2d(self, coords_3d):
        """
        Project 3D point cloud to 2D image

        Args:
            coords_3d: numpy array of shape (N, 3) with normalized coordinates [0, 1]

        Returns:
            image: numpy array of shape (image_size, image_size, 3)
        """
        if self.method == 'density_map':
            return self._density_map_projection(coords_3d)
        elif self.method == 'scatter':
            return self._scatter_projection(coords_3d)
        elif self.method == 'voxel':
            return self._voxel_projection(coords_3d)
        else:
            raise ValueError(f"Unknown projection method: {self.method}")

    def _density_map_projection(self, coords_3d):
        """
        Create density map projection where each pixel encodes:
        - R channel: average X coordinate of nearby points
        - G channel: average Y coordinate of nearby points
        - B channel: average Z coordinate of nearby points
        Pixel intensity represents point density
        """
        image = np.zeros((self.image_size, self.image_size, 3), dtype=np.float32)
        density = np.zeros((self.image_size, self.image_size), dtype=np.float32)

        xy_coords = coords_3d[:, :2]
        scaled_coords = (xy_coords * (self.image_size - 1)).astype(int)

        scaled_coords = np.clip(scaled_coords, 0, self.image_size - 1)

        for i, (x, y) in enumerate(scaled_coords):
            image[y, x] += coords_3d[i]
            density[y, x] += 1
        mask = density > 0
        for c in range(3):
            image[:, :, c][mask] /= density[mask]

        for c in range(3):
            image[:, :, c] = gaussian_filter(image[:, :, c], sigma=1.0)

        return image

    def _scatter_projection(self, coords_3d):
        """
        Direct scatter projection using XY as pixel position and Z as intensity
        RGB = XYZ coordinates
        """
        image = np.zeros((self.image_size, self.image_size, 3), dtype=np.float32)

        xy_coords = coords_3d[:, :2]
        scaled_coords = (xy_coords * (self.image_size - 1)).astype(int)
        scaled_coords = np.clip(scaled_coords, 0, self.image_size - 1)

        for i, (x, y) in enumerate(scaled_coords):
            image[y, x] = coords_3d[i]

        return image

    def _voxel_projection(self, coords_3d):
        """
        Voxel-based projection: divide 3D space into voxels and project
        """

        voxel_size = self.image_size
        voxel_grid = np.zeros((voxel_size, voxel_size, voxel_size), dtype=np.float32)

        voxel_coords = (coords_3d * (voxel_size - 1)).astype(int)
        voxel_coords = np.clip(voxel_coords, 0, voxel_size - 1)

        for coord in voxel_coords:
            voxel_grid[coord[0], coord[1], coord[2]] = 1.0


        image = np.zeros((self.image_size, self.image_size, 3), dtype=np.float32)


        image[:, :, 0] = np.max(voxel_grid, axis=2)

        image[:, :, 1] = np.sum(voxel_grid, axis=2) / voxel_size

        z_coords = np.arange(voxel_size).reshape(1, 1, -1)
        weighted_sum = np.sum(voxel_grid * z_coords, axis=2)
        total = np.sum(voxel_grid, axis=2)
        mask = total > 0
        image[:, :, 2][mask] = weighted_sum[mask] / (total[mask] * voxel_size)

        return image

    def visualize(self, image, save_path=None):
        """
        Visualize the 2D projection

        Args:
            image: numpy array of shape (H, W, 3)
            save_path: Optional path to save the image
        """
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.title('Protein Point Cloud Projection')
        plt.axis('off')

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        else:
            plt.show()

        plt.close()

    def save_image(self, image, save_path):
        """
        Save image as PNG file

        Args:
            image: numpy array of shape (H, W, 3) with values in [0, 1]
            save_path: Path to save the image
        """
        image_uint8 = (image * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_uint8)
        pil_image.save(save_path)


if __name__ == "__main__":
    
    projector = PointCloudProjector(image_size=224, method='density_map')
    test_coords = np.random.rand(1000, 3)
    image = projector.project_to_2d(test_coords)

 
