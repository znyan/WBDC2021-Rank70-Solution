#! /bin/bash


# #################### get env directories
# CONDA_ROOT
CONDA_CONFIG_ROOT_PREFIX=$(conda config --show root_prefix)
echo "CONDA_CONFIG_ROOT_PREFIX= ${CONDA_CONFIG_ROOT_PREFIX}"
get_conda_root_prefix() {
    TMP_POS=$(awk -v a="${CONDA_CONFIG_ROOT_PREFIX}" -v b="/" 'BEGIN{print index(a, b)}')
    TMP_POS=$((TMP_POS-1))
    if [ $TMP_POS -ge 0 ]; then
        echo "${CONDA_CONFIG_ROOT_PREFIX:${TMP_POS}}"
    else
        echo ""
    fi
}
CONDA_ROOT=$(get_conda_root_prefix)
if [ ! -d "${CONDA_ROOT}" ]; then
    echo "CONDA_ROOT= ${CONDA_ROOT}, not exists, exit"
    exit 1
fi

# CONDA ENV
CONDA_NEW_ENV=wbdc

# JUPYTER_ROOT
JUPYTER_ROOT=/home/tione/notebook
if [ ! -d "${JUPYTER_ROOT}" ]; then
    echo "JUPYTER_ROOT= ${JUPYTER_ROOT}, not exists, exit"
    exit 1
fi

# OS RELEASE
OS_ID=$(awk -F= '$1=="ID" { print $2 ;}' /etc/os-release)
OS_ID=${OS_ID//"\""/""}

echo "CONDA_ROOT= ${CONDA_ROOT}"
echo "CONDA_NEW_ENV= ${CONDA_NEW_ENV}"
echo "JUPYTER_ROOT= ${JUPYTER_ROOT}"
echo "OS_ID= ${OS_ID}"

# #################### install or fix environments
ACTION=$(echo "$1" | tr '[:upper:]' '[:lower:]')

echo "[Info] Fix environments"

# #################### add conda env
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda env list
# add envs_dirs
conda config --add envs_dirs ${JUPYTER_ROOT}/envs
conda config --show | grep env
conda env list

exit 0