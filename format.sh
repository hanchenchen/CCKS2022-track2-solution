# remove unused imports
pycln .
# sort import
isort -rc .
# format code
black .
