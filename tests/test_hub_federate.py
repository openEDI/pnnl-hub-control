import json
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest
import xarray as xr

from oedisi.types.common import BrokerConfig
from oedisi.types.data_types import EquipmentNodeArray, MeasurementArray

# Import with helics mocked (conftest patches sys.modules)
from hub_federate import (
    HubFederate,
    StaticConfig,
    Subscriptions,
    eqarray_to_xarray,
    measurement_to_xarray,
    run_simulator,
    xarray_to_dict,
    xarray_to_eqarray,
)


# ---------------------------------------------------------------------------
# Conversion functions
# ---------------------------------------------------------------------------


class TestEqarrayToXarray:
    def test_eqarray_to_xarray(self, sample_equipment_node_array):
        result = eqarray_to_xarray(sample_equipment_node_array)
        assert isinstance(result, xr.DataArray)
        np.testing.assert_array_equal(result.data, [1.0, 2.0])
        assert list(result.coords["ids"].values) == ["a", "b"]
        assert list(result.coords["equipment_ids"].values) == ["eq1", "eq2"]
        assert result.dims == ("ids",)

    def test_eqarray_to_xarray_empty(self):
        eq = EquipmentNodeArray(
            ids=[], equipment_ids=[], values=[], units="MW"
        )
        result = eqarray_to_xarray(eq)
        assert isinstance(result, xr.DataArray)
        assert len(result.data) == 0
        assert "ids" in result.coords
        assert "equipment_ids" in result.coords
        assert result.dims == ("ids",)


class TestMeasurementToXarray:
    def test_measurement_to_xarray(self, sample_measurement_array):
        result = measurement_to_xarray(sample_measurement_array)
        assert isinstance(result, xr.DataArray)
        np.testing.assert_array_equal(result.data, [3.0, 4.0])
        assert list(result.coords["ids"].values) == ["m1", "m2"]

    def test_measurement_to_xarray_empty(self):
        m = MeasurementArray(ids=[], values=[], units="MW")
        result = measurement_to_xarray(m)
        assert isinstance(result, xr.DataArray)
        assert len(result.data) == 0
        assert "ids" in result.coords


class TestXarrayToDict:
    def test_xarray_to_dict(self):
        da = xr.DataArray([1.0, 2.0], coords={"ids": ["a", "b"]})
        result = xarray_to_dict(da)
        assert result == {"values": [1.0, 2.0], "ids": ["a", "b"]}

    def test_xarray_to_dict_with_extra_coords(self):
        da = xr.DataArray(
            [1.0, 2.0],
            dims=("ids",),
            coords={
                "ids": ["a", "b"],
                "equipment_ids": ("ids", ["eq1", "eq2"]),
            },
        )
        result = xarray_to_dict(da)
        assert "ids" in result
        assert "equipment_ids" in result
        assert "values" in result


class TestXarrayToEqarray:
    def test_xarray_to_eqarray(self):
        """Documents current behavior: xarray_to_eqarray is identical to xarray_to_dict."""
        da = xr.DataArray([1.0, 2.0], coords={"ids": ["a", "b"]})
        result = xarray_to_eqarray(da)
        expected = xarray_to_dict(da)
        assert result == expected


class TestXarrayRoundtrip:
    def test_xarray_roundtrip(self, sample_equipment_node_array):
        xa = eqarray_to_xarray(sample_equipment_node_array)
        result = xarray_to_dict(xa)
        assert result["values"] == [1.0, 2.0]
        assert result["ids"] == ["a", "b"]
        assert result["equipment_ids"] == ["eq1", "eq2"]


# ---------------------------------------------------------------------------
# HubFederate — load_static_inputs
# ---------------------------------------------------------------------------


class TestHubFederateLoadStaticInputs:
    def _make_instance(self):
        return object.__new__(HubFederate)

    def test_load_static_inputs_happy_path(self, tmp_path):
        hub = self._make_instance()
        config = {"name": "test_fed", "number_of_timesteps": 10}
        (tmp_path / "static_inputs.json").write_text(json.dumps(config))

        with patch("hub_federate.Path") as MockPath:
            MockPath.return_value.parent = tmp_path
            hub.load_static_inputs()

        assert hub.static.name == "test_fed"
        assert hub.static.t_steps == 10

    def test_load_static_inputs_missing_file(self, tmp_path):
        hub = self._make_instance()
        with patch("hub_federate.Path") as MockPath:
            MockPath.return_value.parent = tmp_path
            with pytest.raises(FileNotFoundError):
                hub.load_static_inputs()

    def test_load_static_inputs_missing_keys(self, tmp_path):
        hub = self._make_instance()
        (tmp_path / "static_inputs.json").write_text("{}")
        with patch("hub_federate.Path") as MockPath:
            MockPath.return_value.parent = tmp_path
            with pytest.raises(KeyError):
                hub.load_static_inputs()


# ---------------------------------------------------------------------------
# HubFederate — load_input_mapping
# ---------------------------------------------------------------------------


class TestHubFederateLoadInputMapping:
    def _make_instance(self):
        return object.__new__(HubFederate)

    def test_load_input_mapping_happy_path(self, tmp_path, hub_input_mapping):
        hub = self._make_instance()
        (tmp_path / "input_mapping.json").write_text(json.dumps(hub_input_mapping))

        with patch("hub_federate.Path") as MockPath:
            MockPath.return_value.parent = tmp_path
            hub.load_input_mapping()

        assert hub.inputs == hub_input_mapping

    def test_load_input_mapping_missing_file(self, tmp_path):
        hub = self._make_instance()
        with patch("hub_federate.Path") as MockPath:
            MockPath.return_value.parent = tmp_path
            with pytest.raises(FileNotFoundError):
                hub.load_input_mapping()


# ---------------------------------------------------------------------------
# HubFederate — load_component_definition
# ---------------------------------------------------------------------------


class TestHubFederateLoadComponentDefinition:
    def test_load_component_definition_happy_path(self, tmp_path):
        hub = object.__new__(HubFederate)
        config = {"type": "hub_control", "version": "1.0"}
        (tmp_path / "component_definition.json").write_text(json.dumps(config))

        with patch("hub_federate.Path") as MockPath:
            MockPath.return_value.parent = tmp_path
            hub.load_component_definition()

        assert hub.component_config == config


# ---------------------------------------------------------------------------
# HubFederate — initialize
# ---------------------------------------------------------------------------


class TestHubFederateInitialize:
    def test_initialize_creates_federate(self, mock_helics):
        hub = object.__new__(HubFederate)
        hub.static = StaticConfig()
        hub.static.name = "test_fed"

        import hub_federate

        with patch.object(hub_federate, "h", mock_helics):
            hub.initilize(BrokerConfig(broker_ip="127.0.0.1", broker_port=23404))

        mock_helics.helicsCreateFederateInfo.assert_called_once()
        mock_helics.helicsFederateInfoSetBroker.assert_called_with(
            mock_helics.helicsCreateFederateInfo.return_value, "127.0.0.1"
        )
        mock_helics.helicsFederateInfoSetBrokerPort.assert_called_with(
            mock_helics.helicsCreateFederateInfo.return_value, 23404
        )
        mock_helics.helicsCreateValueFederate.assert_called_with(
            "test_fed", mock_helics.helicsCreateFederateInfo.return_value
        )
        mock_helics.helicsFederateSetTimeProperty.assert_called_with(
            mock_helics.helicsCreateValueFederate.return_value,
            mock_helics.HELICS_PROPERTY_TIME_PERIOD,
            1,
        )


# ---------------------------------------------------------------------------
# HubFederate — register_subscription
# ---------------------------------------------------------------------------


class TestHubFederateRegisterSubscription:
    def test_register_subscription_all_five(self, hub_input_mapping):
        hub = object.__new__(HubFederate)
        hub.sub = Subscriptions()
        hub.fed = MagicMock()
        hub.inputs = hub_input_mapping

        hub.register_subscription()

        assert hub.fed.register_subscription.call_count == 5
        for key in ("sub_c0", "sub_c1", "sub_c2", "sub_c3", "sub_c4"):
            hub.fed.register_subscription.assert_any_call(
                hub_input_mapping[key], ""
            )


# ---------------------------------------------------------------------------
# HubFederate — register_publication
# ---------------------------------------------------------------------------


class TestHubFederateRegisterPublication:
    def test_register_publication(self, mock_helics):
        hub = object.__new__(HubFederate)
        hub.fed = MagicMock()

        import hub_federate

        with patch.object(hub_federate, "h", mock_helics):
            hub.register_publication()

        hub.fed.register_publication.assert_called_once_with(
            "pv_set", mock_helics.HELICS_DATA_TYPE_STRING, ""
        )


# ---------------------------------------------------------------------------
# HubFederate — run
# ---------------------------------------------------------------------------


class TestHubFederateRun:
    def _setup_hub(self, mock_h, sub_configs):
        """Create a HubFederate with mocked subscriptions.

        sub_configs: dict mapping c0-c4 to (is_updated: bool, json_data: list)
        """
        hub = object.__new__(HubFederate)
        hub.static = StaticConfig()
        hub.static.t_steps = 0
        hub.sub = Subscriptions()
        hub.fed = MagicMock()
        hub.pub_commands = MagicMock()

        for name in ("c0", "c1", "c2", "c3", "c4"):
            sub = MagicMock()
            updated, data = sub_configs.get(name, (False, []))
            sub.is_updated.return_value = updated
            sub.json = data
            setattr(hub.sub, name, sub)

        return hub

    def test_run_single_iteration_all_updated(self, mock_helics):
        import hub_federate

        command_data = [{"id": "x", "value": 1}]
        sub_configs = {
            f"c{i}": (True, command_data) for i in range(5)
        }

        # Grant time 1 first, then 2 (exceeds t_steps=1 on next check)
        granted_times = iter([1, 2])

        with patch.object(hub_federate, "h", mock_helics):
            mock_helics.helicsFederateGetTimeProperty.return_value = 1.0
            mock_helics.helicsFederateRequestTime.side_effect = lambda fed, t: next(granted_times)

            hub = self._setup_hub(mock_helics, sub_configs)
            hub.stop = MagicMock()
            hub.run()

        # Should have published once with 5 commands (one from each area)
        assert hub.pub_commands.publish.call_count == 1
        published = json.loads(hub.pub_commands.publish.call_args[0][0])
        assert len(published) == 5

        mock_helics.helicsFederateEnterExecutingMode.assert_called_once()

    def test_run_single_iteration_partial_update(self, mock_helics):
        import hub_federate

        sub_configs = {
            "c0": (True, [{"id": "x", "value": 1}]),
            "c1": (False, []),
            "c2": (False, []),
            "c3": (True, [{"id": "y", "value": 2}]),
            "c4": (False, []),
        }

        granted_times = iter([1, 2])

        with patch.object(hub_federate, "h", mock_helics):
            mock_helics.helicsFederateGetTimeProperty.return_value = 1.0
            mock_helics.helicsFederateRequestTime.side_effect = lambda fed, t: next(granted_times)

            hub = self._setup_hub(mock_helics, sub_configs)
            hub.stop = MagicMock()
            hub.run()

        published = json.loads(hub.pub_commands.publish.call_args[0][0])
        assert len(published) == 2

    def test_run_single_iteration_no_updates(self, mock_helics):
        import hub_federate

        sub_configs = {f"c{i}": (False, []) for i in range(5)}

        granted_times = iter([1, 2])

        with patch.object(hub_federate, "h", mock_helics):
            mock_helics.helicsFederateGetTimeProperty.return_value = 1.0
            mock_helics.helicsFederateRequestTime.side_effect = lambda fed, t: next(granted_times)

            hub = self._setup_hub(mock_helics, sub_configs)
            hub.stop = MagicMock()
            hub.run()

        published = hub.pub_commands.publish.call_args[0][0]
        assert published == "[]"

    def test_run_calls_stop_on_completion(self, mock_helics):
        import hub_federate

        sub_configs = {f"c{i}": (False, []) for i in range(5)}
        granted_times = iter([1, 2])

        with patch.object(hub_federate, "h", mock_helics):
            mock_helics.helicsFederateGetTimeProperty.return_value = 1.0
            mock_helics.helicsFederateRequestTime.side_effect = lambda fed, t: next(granted_times)

            hub = self._setup_hub(mock_helics, sub_configs)
            hub.stop = MagicMock()
            hub.run()

        hub.stop.assert_called_once()


# ---------------------------------------------------------------------------
# HubFederate — stop
# ---------------------------------------------------------------------------


class TestHubFederateStop:
    def test_stop_disconnects_and_frees(self, mock_helics):
        import hub_federate

        hub = object.__new__(HubFederate)
        hub.fed = MagicMock()

        with patch.object(hub_federate, "h", mock_helics):
            hub.stop()

        mock_helics.helicsFederateDisconnect.assert_called_once_with(hub.fed)
        mock_helics.helicsFederateFree.assert_called_once_with(hub.fed)
        mock_helics.helicsCloseLibrary.assert_called_once()

        # Verify order: disconnect before free before close
        calls = mock_helics.method_calls
        disconnect_idx = next(
            i for i, c in enumerate(calls) if c[0] == "helicsFederateDisconnect"
        )
        free_idx = next(
            i for i, c in enumerate(calls) if c[0] == "helicsFederateFree"
        )
        close_idx = next(
            i for i, c in enumerate(calls) if c[0] == "helicsCloseLibrary"
        )
        assert disconnect_idx < free_idx < close_idx


# ---------------------------------------------------------------------------
# run_simulator
# ---------------------------------------------------------------------------


class TestRunSimulator:
    def test_run_simulator(self, mock_helics, tmp_path):
        import hub_federate

        # Write config files needed by __init__
        static = {"name": "test_hub", "number_of_timesteps": 5}
        inputs = {
            "sub_c0": "a/p0", "sub_c1": "a/p1", "sub_c2": "a/p2",
            "sub_c3": "a/p3", "sub_c4": "a/p4",
        }
        comp_def = {"type": "hub_control"}
        (tmp_path / "static_inputs.json").write_text(json.dumps(static))
        (tmp_path / "input_mapping.json").write_text(json.dumps(inputs))
        (tmp_path / "component_definition.json").write_text(json.dumps(comp_def))

        with patch.object(hub_federate, "h", mock_helics), \
             patch("hub_federate.Path") as MockPath:
            MockPath.return_value.parent = tmp_path

            mock_fed_instance = MagicMock()
            with patch.object(hub_federate, "HubFederate", wraps=None) as MockClass:
                MockClass.return_value = mock_fed_instance
                broker = BrokerConfig(broker_ip="0.0.0.0")
                run_simulator(broker)

                MockClass.assert_called_once_with(broker)
                mock_fed_instance.run.assert_called_once()
