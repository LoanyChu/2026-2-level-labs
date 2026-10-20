set -ex

echo $1
if [[ "$1" == "smoke" ]]; then
  DIRS_TO_CHECK=(
    "admin_utils"
    "seminars"
    "lab_1_classify_profile"
    "lab_2_tokenize_by_bpe"
    "lab_3_generate_by_ngrams"
  )
else
  DIRS_TO_CHECK=(
    "admin_utils"
    "seminars"
    "lab_1_classify_profile"
    "lab_2_tokenize_by_bpe"
    "lab_3_generate_by_ngrams"
  )
fi

source venv/bin/activate
export PYTHONPATH=$(pwd)

fiplconfig.generate_labs_stubs

python -m black "${DIRS_TO_CHECK[@]}"

isort .

python -m pylint "${DIRS_TO_CHECK[@]}"

mypy "${DIRS_TO_CHECK[@]}"

python -m flake8 "${DIRS_TO_CHECK[@]}"

pydoctest --config pydoctest.json


if [[ "$1" != "smoke" ]]; then
  fiplconfig.check_doc8

  fiplconfig.check_spelling

  rm -rf dist
  sphinx-build -b html -W --keep-going -n . dist -c admin_utils

  python -m pytest -m "mark10 and lab_1_classify_profile"
  python -m pytest -m "mark10 and lab_2_tokenize_by_bpe"
  python -m pytest -m "mark10 and lab_3_generate_by_ngrams"
fi
