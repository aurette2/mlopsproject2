import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from model import Unet
import os

# Constants for the tests
IMG_SIZE = 128
NUM_CLASSES = 4
MODEL_PATH = os.path.join("models", "my_model.keras")
VOLUME_SLICES = 100

@pytest.fixture
def mock_unet():
    # Initialize the Unet model and mock the compile_and_load_weights method
    model = Unet(img_size=IMG_SIZE, num_classes=NUM_CLASSES)
    model.compile_and_load_weights = MagicMock()
    model.model = MagicMock()  # Mock the Keras model
    return model

def test_build_model(mock_unet):
    model = mock_unet.build_model()
    assert model.input_shape == (None, IMG_SIZE, IMG_SIZE, 2)
    assert model.output_shape == (None, IMG_SIZE, IMG_SIZE, NUM_CLASSES)
    assert len(model.layers) > 0  # Ensure layers were added to the model

def test_compile_model(mock_unet):
    mock_unet.compile_model()
    mock_unet.model.compile.assert_called_once()  # Ensure model.compile() was called once

def test_predictFromFiles(mock_unet):
    # Mock the nibabel and cv2 loading functions to return a fixed array
    with patch("model.nib.load") as mock_nib_load, patch("model.cv2.resize", return_value=np.ones((IMG_SIZE, IMG_SIZE))):
        mock_nib_load.return_value.get_fdata.return_value = np.ones((240, 240, VOLUME_SLICES + 22))
        mock_unet.model.predict.return_value = np.random.rand(VOLUME_SLICES, IMG_SIZE, IMG_SIZE, NUM_CLASSES)

        flair_path = "mock_flair.nii"
        t1ce_path = "mock_t1ce.nii"
        prediction = mock_unet.predictFromFiles(flair_path, t1ce_path)

        assert prediction.shape == (VOLUME_SLICES, IMG_SIZE, IMG_SIZE, NUM_CLASSES)
        assert mock_unet.model.predict.called

def test_evaluate_model(mock_unet):
    # Mock the evaluate method
    mock_unet.model.evaluate.return_value = [0.2, 0.9, 0.8, 0.85, 0.87, 0.9, 0.89, 0.8, 0.82]

    mock_test_generator = MagicMock()  # Mock test generator
    metrics = mock_unet.evaluate(mock_test_generator)

    assert isinstance(metrics, dict)
    assert "Accuracy" in metrics
    assert metrics["Accuracy"] == 0.9  # Checking the accuracy

def test_imageLoader(mock_unet):
    # Mock the nibabel and cv2 loading functions to return a fixed array
    with patch("model.nib.load") as mock_nib_load, patch("model.cv2.resize", return_value=np.ones((IMG_SIZE, IMG_SIZE))):
        mock_nib_load.return_value.get_fdata.return_value = np.ones((240, 240, VOLUME_SLICES + 22))

        path = "mock_path.nii"
        result = mock_unet.imageLoader(path)

        assert result.shape == (VOLUME_SLICES, IMG_SIZE, IMG_SIZE, 2)
        assert np.all(result >= 0)  # Ensures that all values are valid

def test_showPredictsFromFile(mock_unet):
    # Generate a mock prediction and original image
    prediction = np.random.rand(VOLUME_SLICES, IMG_SIZE, IMG_SIZE, NUM_CLASSES)
    orig_image = np.random.rand(240, 240, VOLUME_SLICES + 22)

    with patch("model.plt.show") as mock_show:
        mock_unet.showPredictsFromFile(prediction, orig_image, start_slice=60)
        mock_show.assert_called_once()

def test_predict_segmentation(mock_unet):
    # Mock the nibabel and cv2 loading functions
    with patch("model.nib.load") as mock_nib_load, patch("model.cv2.resize", return_value=np.ones((IMG_SIZE, IMG_SIZE))):
        mock_nib_load.return_value.get_fdata.return_value = np.ones((240, 240, VOLUME_SLICES + 22))
        mock_unet.model.predict.return_value = np.random.rand(VOLUME_SLICES, IMG_SIZE, IMG_SIZE, NUM_CLASSES)

        flair_path = "mock_flair.nii"
        t1ce_path = "mock_t1ce.nii"
        prediction = mock_unet.predict_segmentation(flair_path, t1ce_path)

        assert prediction.shape == (VOLUME_SLICES, IMG_SIZE, IMG_SIZE, NUM_CLASSES)
        assert mock_unet.model.predict.called