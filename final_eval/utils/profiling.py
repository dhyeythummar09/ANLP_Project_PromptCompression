"""CLI process measurements, including cold loads; no synthetic latency claims."""
import sys
import time
from pathlib import Path
from final_eval.utils.artifacts import atomic_json


def profile_cli(function, output_flag, default=None, directory=False):
    if "--dry-run" in sys.argv:
        return function()
    output = default
    for i, value in enumerate(sys.argv[1:], 1):
        if value == output_flag and i+1 < len(sys.argv):
            output = sys.argv[i+1]
        elif value.startswith(output_flag+"="):
            output = value.split("=",1)[1]
    start=time.perf_counter(); outcome="completed"
    try:
        return function()
    except BaseException:
        outcome="interrupted_or_failed"
        raise
    finally:
        if output:
            target=Path(output)
            # A rejected invocation must not create a result directory.
            if target.exists() or Path(str(target)+".manifest.json").exists():
                record={"status":outcome,"wall_seconds":time.perf_counter()-start,
                        "scope":"One CLI process, including cold model loads. Resumed invocations have separate records; response times can be shared by cached conditions.",
                        "cpu_peak_rss_bytes":None,"cuda":[]}
                try:
                    import resource
                    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    record["cpu_peak_rss_bytes"]=peak if sys.platform=="darwin" else peak*1024
                except ImportError:
                    pass
                torch=sys.modules.get("torch")
                if torch is not None and torch.cuda.is_initialized():
                    for i in range(torch.cuda.device_count()):
                        peak=torch.cuda.max_memory_allocated(i)
                        if peak:
                            record["cuda"].append({"device":torch.cuda.get_device_name(i),
                                "peak_allocated_bytes":peak,"peak_reserved_bytes":torch.cuda.max_memory_reserved(i)})
                folder=target/"profiles" if directory else Path(str(target)+".profiles")
                atomic_json(folder/f"process_{time.time_ns()}.json",record)
