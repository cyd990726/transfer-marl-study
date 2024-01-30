REGISTRY = {}

# normal controllers
from .basic_controller import BasicMAC
from .basic_dc_controller import BasicDCMAC
from .trans_controller import TransMAC
from .decomposed_controller import DecomposedMAC
from .xtrans_controller import XTransMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["basic_dc_mac"] = BasicDCMAC
REGISTRY["decomposed_mac"] = DecomposedMAC
REGISTRY["trans_mac"] = TransMAC
REGISTRY["xtrans_mac"] = XTransMAC


# some mutli-task controllers
from .multi_task import TransMAC as MultiTaskTransMAC
from .multi_task import XTransMAC as MultiTaskXTransMAC

REGISTRY["mt_trans_mac"] = MultiTaskTransMAC
REGISTRY["mt_xtrans_mac"] = MultiTaskXTransMAC
