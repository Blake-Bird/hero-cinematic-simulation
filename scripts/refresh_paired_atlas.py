from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core.paired_atlas_site import refresh_site
refresh_site(Path(sys.argv[1] if len(sys.argv)>1 else 'N128_EDE_LCDM_Comparison'))
