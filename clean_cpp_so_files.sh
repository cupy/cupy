
# 获取脚本所在目录的绝对路径
SCRIPT_DIR=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
echo "script folder path: $SCRIPT_DIR"

# 尝试列出配置目录（如果存在）
if [ -d "${SCRIPT_DIR}/cupy/_core" ]; then
    ls -l "${SCRIPT_DIR}"
else
    echo "Warning: this script is not root folder of numpy-ascend/cupy repo, skip cleaning files"
    exit -1
fi

# clean up so file ,to switch backend
NUMPY_ASCEND_DIR=${SCRIPT_DIR}
cd ${NUMPY_ASCEND_DIR}/cupy/_core && rm *.cpp && rm *.so 
cd ${NUMPY_ASCEND_DIR}/cupy/xpu && rm *.cpp && rm *.so
cd ${NUMPY_ASCEND_DIR}/cupy/backends/backend/api && rm *.cpp && rm *.so 
cd ${NUMPY_ASCEND_DIR}/cupy/backends/backend && rm *.cpp && rm *.so
cd ${NUMPY_ASCEND_DIR}/cupy/backends/ascend/api && rm *.cpp && rm *.so

cd ${NUMPY_ASCEND_DIR}
echo "Sucessfully clean cython generated cpp and so files"