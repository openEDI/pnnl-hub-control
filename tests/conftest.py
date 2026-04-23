import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add source directory to path so hub_federate and server can be imported directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "pnnl-hub-control"))

from oedisi.types.data_types import EquipmentNodeArray, MeasurementArray


@pytest.fixture(scope="session")
def mock_helics():
    """Session-scoped fixture that patches the helics module with a MagicMock."""
    mock_h = MagicMock()

    # Constants
    mock_h.HELICS_CORE_TYPE_ZMQ = 2
    mock_h.HELICS_DATA_TYPE_STRING = "string"
    mock_h.HELICS_PROPERTY_TIME_PERIOD = 140

    # helicsCreateFederateInfo returns a mock info object
    mock_info = MagicMock()
    mock_h.helicsCreateFederateInfo.return_value = mock_info

    # helicsCreateValueFederate returns a mock federate
    mock_fed = MagicMock()
    mock_fed.register_subscription = MagicMock()
    mock_fed.register_publication = MagicMock()
    mock_h.helicsCreateValueFederate.return_value = mock_fed

    # helicsFederateRequestTime: configurable sequence
    mock_h.helicsFederateRequestTime.return_value = 1

    # helicsFederateGetTimeProperty returns 1.0 (period)
    mock_h.helicsFederateGetTimeProperty.return_value = 1.0

    # Patch helics in sys.modules
    sys.modules["helics"] = mock_h

    yield mock_h

    # Restore
    if "helics" in sys.modules and sys.modules["helics"] is mock_h:
        del sys.modules["helics"]


@pytest.fixture
def hub_static_config():
    """Returns a dict suitable for writing to static_inputs.json."""
    return {"name": "test_hub", "number_of_timesteps": 5}


@pytest.fixture
def hub_input_mapping():
    """Returns a dict suitable for writing to input_mapping.json."""
    return {
        "sub_c0": "src0/port0",
        "sub_c1": "src1/port1",
        "sub_c2": "src2/port2",
        "sub_c3": "src3/port3",
        "sub_c4": "src4/port4",
    }


@pytest.fixture
def sample_equipment_node_array():
    return EquipmentNodeArray(
        ids=["a", "b"],
        equipment_ids=["eq1", "eq2"],
        values=[1.0, 2.0],
        units="MW",
    )


@pytest.fixture
def sample_measurement_array():
    return MeasurementArray(
        ids=["m1", "m2"],
        values=[3.0, 4.0],
        units="MW",
    )
