import sys
sys.path.append('/project/project_462000451/enchanted-surrogates_11feb2025/src')
from parsers.GENEparser import GENEparser

print('STARTING HELENA TO EQDISK CONVERSION')
print(sys.argv)

this_script_path, fpath, eqdsk_path = sys.argv

gp = GENEparser()
NR = 900
NZ = 900

print('showing fpath and eqdsk path in python')
print(fpath)
print(eqdsk_path)

eqdsk_out = gp.helena_to_eqdsk(fpath, eqdsk_path, NR, NZ)