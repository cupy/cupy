
# build CANN extra ops packages
```bash
# aussuming all ops repos has been clone
# git clone -b ${tag_version} https://gitcode.com/cann/ops-sparse.git
cd ../repos/
cd ops-sparse/

git pull oriign master

# check and make sure 
echo $ASCEND_HOME_PATH

# ops-sparse has a special option:  --ops=spmv
bash build.sh --pkg --soc=ascend950 bash build.sh --ops=coo2csr,coo_get,coosort,cscsort,csr2coo,csr2csc_ex2,csr2gebsr,\
csrgeam2,csrsort,cube_spmm,densetosparse,gather,gebsr2gebsc,gpsv_interleaved_batch,\
gtsv2,gtsv2_nopivot,gtsv2_strided_batch,gtsv_interleaved_batch,nnz,scatter,sddmm,\
spgemm,spmm,spmm_op,spmv,spmv_op,spsm,spsv,spvv

# 默认 = --soc=ascend910b3，全量算子
# -> sparse/sparse2dense/arch22 编不过（op_host 引用 device-only 的 __aicore__），整次构建中断


# if fine, sh file will be generated
./build_out/cann-*-ops-sparse-*.run --install --install-path="$ASCEND_HOME_PATH"

```

[OpsSparse] [2026-10-17 11:27:10] [INFO]: Installing ops_sparse to /home/qingfeng/miniconda3/envs/aigent/Ascend/cann-9.0.1/cann
[OpsSparse] [2026-10-17 11:27:46] [INFO]: ops_sparse installed successfully to /home/qingfeng/miniconda3/envs/aigent/Ascend/cann-9.0.1/cann
[OpsSparse] [2026-10-17 11:27:46] [INFO]: Libraries: /home/qingfeng/miniconda3/envs/aigent/Ascend/cann-9.0.1/cann/lib64/ (1 .so files)
[OpsSparse] [2026-10-17 11:27:46] [INFO]: Headers:   /home/qingfeng/miniconda3/envs/aigent/Ascend/cann-9.0.1/cann/include/

###`libops_sparse.so`  so file install path

$ASCEND_HOME_PATH=/home/qingfeng/miniconda3/envs/aigent/Ascend/cann-9.0.1，
所以是 .../cann-9.0.1/cann/x86_64-linux/lib64/libops_sparse.so —— 注意不在 $ASCEND_HOME_PATH/lib64，

可以追加这个LD路径
```bash
. "$HOME/.load_cann90.sh"
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/cann/lib64:$LD_LIBRARY_PATH
```

This is similar for other `ops-xxx`