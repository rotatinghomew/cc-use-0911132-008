set -eu
workspace=$(mktemp -d)
trap 'rm -rf "$workspace"' EXIT
export PYTHONDONTWRITEBYTECODE=1
python3 -m unittest discover -s tests/rights_cases -p 'test_*.py' -q
python3 -m unittest discover -s tests -p test_analysis_inputs.py -v
PYTHONPYCACHEPREFIX="$workspace/cache" python3 -m compileall -q design_rights_chain tests
python3 -m design_rights_chain.casework.command fixtures/demo.json --database "$workspace/operations.sqlite3" > "$workspace/operations.json"
python3 -m json.tool "$workspace/operations.json"
