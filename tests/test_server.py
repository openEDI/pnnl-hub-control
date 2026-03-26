import json
import os
import stat
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from server import app, find_filenames


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def tmp_cwd(tmp_path, monkeypatch):
    """Change working directory to tmp_path for tests that touch the filesystem."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------


class TestReadRoot:
    def test_read_root(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "hostname" in data
        assert "host_ip" in data
        assert isinstance(data["hostname"], str)
        assert isinstance(data["host_ip"], str)
        assert len(data["hostname"]) > 0
        assert len(data["host_ip"]) > 0


# ---------------------------------------------------------------------------
# find_filenames
# ---------------------------------------------------------------------------


class TestFindFilenames:
    def test_find_filenames_with_feather_files(self, tmp_path):
        (tmp_path / "data.feather").write_bytes(b"fake")
        (tmp_path / "other.csv").write_text("a,b")
        result = find_filenames(str(tmp_path))
        assert result == ["data.feather"]

    def test_find_filenames_no_matches(self, tmp_path):
        (tmp_path / "other.csv").write_text("a,b")
        result = find_filenames(str(tmp_path))
        assert result == []

    def test_find_filenames_custom_suffix(self, tmp_path):
        (tmp_path / "file.json").write_text("{}")
        result = find_filenames(str(tmp_path), suffix=".json")
        assert result == ["file.json"]

    def test_find_filenames_empty_directory(self, tmp_path):
        result = find_filenames(str(tmp_path))
        assert result == []


# ---------------------------------------------------------------------------
# GET /download
# ---------------------------------------------------------------------------


class TestDownloadResults:
    def test_download_results_found(self, client, tmp_cwd):
        content = b"feather-binary-content"
        (tmp_cwd / "results.feather").write_bytes(content)
        # find_filenames default arg is frozen at import time, so patch it
        with patch("server.find_filenames", return_value=["results.feather"]):
            response = client.get("/download")
        assert response.status_code == 200
        assert response.content == content

    def test_download_results_not_found(self, client, tmp_cwd):
        response = client.get("/download")
        assert response.status_code == 404
        assert response.json()["detail"] == "No feather file found"


# ---------------------------------------------------------------------------
# POST /run
# ---------------------------------------------------------------------------


class TestRunModel:
    def test_run_model_success(self, client):
        with patch("server.run_simulator"):
            response = client.post(
                "/run",
                json={"broker_ip": "127.0.0.1", "broker_port": 23404},
            )
        assert response.status_code == 200
        assert response.json()["detail"] == "Task sucessfully added."

    def test_run_model_invalid_body(self, client):
        # BrokerConfig has defaults for all fields, so {} is valid.
        # Send a non-coercible value to trigger validation error.
        response = client.post("/run", json={"broker_port": "not_a_number"})
        assert response.status_code == 422

    def test_run_model_exception_swallowed(self, client):
        """Documents current buggy behavior: missing 'raise' on line 60.

        The except block creates HTTPException but doesn't raise it,
        so the function returns None and FastAPI returns 200 with null body.
        """
        with patch("server.BackgroundTasks.add_task", side_effect=RuntimeError("boom")):
            response = client.post(
                "/run",
                json={"broker_ip": "127.0.0.1", "broker_port": 23404},
            )
        assert response.status_code == 200
        assert response.json() is None


# ---------------------------------------------------------------------------
# POST /configure
# ---------------------------------------------------------------------------


class TestConfigure:
    def _make_component_struct(self, links=None, parameters=None, name="test_comp"):
        """Build a valid ComponentStruct payload."""
        if parameters is None:
            parameters = {"param1": "value1"}
        if links is None:
            links = [
                {
                    "source": "source_comp",
                    "source_port": "out_port",
                    "target": "target_comp",
                    "target_port": "sub_c0",
                }
            ]
        return {
            "component": {
                "name": name,
                "type": "hub_control",
                "parameters": parameters,
            },
            "links": links,
        }

    def test_configure_success(self, client, tmp_cwd):
        payload = self._make_component_struct()
        response = client.post("/configure", json=payload)
        assert response.status_code == 200

        # Verify input_mapping.json
        with open(tmp_cwd / "input_mapping.json") as f:
            mapping = json.load(f)
        assert mapping == {"sub_c0": "source_comp/out_port"}

        # Verify static_inputs.json
        with open(tmp_cwd / "static_inputs.json") as f:
            static = json.load(f)
        assert static["name"] == "test_comp"
        assert static["param1"] == "value1"

    def test_configure_writes_correct_link_mapping(self, client, tmp_cwd):
        links = [
            {
                "source": "comp_a",
                "source_port": "port_a",
                "target": "hub",
                "target_port": "sub_c0",
            },
            {
                "source": "comp_b",
                "source_port": "port_b",
                "target": "hub",
                "target_port": "sub_c1",
            },
            {
                "source": "comp_c",
                "source_port": "port_c",
                "target": "hub",
                "target_port": "sub_c2",
            },
        ]
        payload = self._make_component_struct(links=links)
        response = client.post("/configure", json=payload)
        assert response.status_code == 200

        with open(tmp_cwd / "input_mapping.json") as f:
            mapping = json.load(f)
        assert mapping == {
            "sub_c0": "comp_a/port_a",
            "sub_c1": "comp_b/port_b",
            "sub_c2": "comp_c/port_c",
        }

    def test_configure_invalid_body(self, client):
        response = client.post("/configure", json={"bad": "data"})
        assert response.status_code == 422

    def test_configure_write_permission_error(self, tmp_cwd):
        # Use a separate client that doesn't raise server exceptions
        error_client = TestClient(app, raise_server_exceptions=False)
        # Make tmp_cwd read-only
        tmp_cwd.chmod(stat.S_IRUSR | stat.S_IXUSR)
        try:
            payload = self._make_component_struct()
            response = error_client.post("/configure", json=payload)
            assert response.status_code == 500
        finally:
            # Restore permissions so pytest can clean up
            tmp_cwd.chmod(stat.S_IRWXU)
