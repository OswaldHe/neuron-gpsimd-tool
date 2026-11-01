# usage: mk_customop_hlo3.py out.pb lib_path function_name
import sys, hashlib
from jax._src.lib import xla_client as xc
out, lib, fn = sys.argv[1:4]
md5 = hashlib.md5(open(lib, "rb").read()).hexdigest()
cfg = (f"version=1.0;name={fn};path={lib};hash={md5};"
       f"ulib_to_ucode_version=1.25.36.0;ulib_to_isa_version=1.0.2520.0;")
txt = f'''HloModule cop, entry_computation_layout={{(f32[128,64]{{1,0}})->f32[128,64]{{1,0}}}}

ENTRY main {{
  p0 = f32[128,64]{{1,0}} parameter(0)
  ROOT r = f32[128,64]{{1,0}} custom-call(p0), custom_call_target="AwsNeuronCustomOp", backend_config="{cfg}"
}}
'''
open(out + ".hlo", "w").write(txt)
open(out, "wb").write(xc._xla.hlo_module_from_text(txt).as_serialized_hlo_module_proto())
