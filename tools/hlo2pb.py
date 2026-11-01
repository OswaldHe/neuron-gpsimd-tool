import sys
from jax._src.lib import xla_client as xc
m = xc._xla.hlo_module_from_text(open(sys.argv[1]).read())
open(sys.argv[2], "wb").write(m.as_serialized_hlo_module_proto())
