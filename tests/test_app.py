import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from main import app  # Replace 'main' with the filename of your FastAPI application script
import io

# Instantiate the test client
client = TestClient(app)

@pytest.fixture
def mock_auth():
    with patch("main.decode_token") as mock_decode:
        mock_decode.return_value = {"sub": "test_user", "role": "admin"}
        yield mock_decode

@pytest.fixture
def mock_auth_user():
    with patch("main.decode_token") as mock_decode:
        mock_decode.return_value = {"sub": "test_user", "role": "user"}
        yield mock_decode


def test_get_hello():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome"}

def test_login_success():
    with patch("main.authenticate_user") as mock_authenticate_user, patch("main.create_access_token") as mock_create_token:
        mock_authenticate_user.return_value = {"username": "test_user", "role": "admin"}
        mock_create_token.return_value = "test_token"

        response = client.post("/token", data={"username": "admin", "password": "adminpass"})
        assert response.status_code == 200
        assert "access_token" in response.json()


def test_read_users_me(mock_auth):
    response = client.get("/users/me/", headers={"Authorization": "Bearer testtoken"})
    assert response.status_code == 200
    assert response.json() == {"sub": "test_user", "role": "admin"}

def test_get_case():
    response = client.get("/case")
    assert response.status_code == 200
    assert isinstance(response.json(), str)  # Adjust this as per your actual response type


def test_get_samples_list():
    response = client.get("/samples_list")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_evaluate_model_api(mock_auth):
    with patch("main.unet_model.evaluate") as mock_evaluate:
        mock_evaluate.return_value = {"Accuracy": 0.9, "MeanIOU": 0.8}
        response = client.post("/evaluate/", headers={"Authorization": "Bearer testtoken"})
        assert response.status_code == 200
        assert response.json() == {"Accuracy": 0.9, "MeanIOU": 0.8}


def test_show_drift_admin(mock_auth):
    with patch("main.generate_drift_report") as mock_generate_report:
        mock_generate_report.return_value = "<html>Drift report content</html>"
        response = client.get("/showdrift/", headers={"Authorization": "Bearer testtoken"})
        assert response.status_code == 200
        assert "Drift report content" in response.text


def test_predictbypath(mock_auth):
    test_flair = io.BytesIO(b"fake flair data")
    test_t1ce = io.BytesIO(b"fake t1ce data")
    
    files = {
        "flair": ("flair.nii", test_flair, "application/octet-stream"),
        "t1ce": ("t1ce.nii", test_t1ce, "application/octet-stream"),
    }

    with patch("main.unet_model.predictFromFiles") as mock_predict:
        mock_predict.return_value = [[0, 1, 2], [3, 4, 5]]
        response = client.post("/predictbypath/", files=files)
        assert response.status_code == 200
        assert response.json() == {"prediction": [[0, 1, 2], [3, 4, 5]]}


def test_show_predicted_segmentations_api(mock_auth):
    test_flair = io.BytesIO(b"fake flair data")
    test_t1ce = io.BytesIO(b"fake t1ce data")

    files = {
        "flair": ("flair.nii", test_flair, "application/octet-stream"),
        "t1ce": ("t1ce.nii", test_t1ce, "application/octet-stream"),
    }

    with patch("main.unet_model.predict_segmentation") as mock_predict_segmentation:
        mock_predict_segmentation.return_value = [[0, 1, 2], [3, 4, 5]]
        response = client.post("/showPredictSegmented/", files=files)
        assert response.status_code == 200
        assert response.json() == {"prediction": [[0, 1, 2], [3, 4, 5]]}