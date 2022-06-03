use libc::{c_float, c_uint};
use ndarray::arr2;
use ndarray::ArrayBase;
use ndarray::Dim;
use ndarray::OwnedRepr;
use std::convert::TryInto;
use std::os::raw::c_void;
use std::os::unix::ffi::OsStrExt;
use std::{ffi, path::Path, ptr, slice};

use xgboost_bib::*;

use super::{XGBError, XGBResult};

static KEY_GROUP_PTR: &'static str = "group_ptr";
static KEY_GROUP: &'static str = "group";
static KEY_LABEL: &'static str = "label";
static KEY_WEIGHT: &'static str = "weight";
static KEY_BASE_MARGIN: &'static str = "base_margin";

#[derive(Debug)]
#[repr(C)]
pub struct DIter {
    pub(super) handle: xgboost_bib::DataIterHandle,
    n: usize,
    lengths: usize,
    cur_it: usize,
    proxy: *mut ffi::c_void,
    data: [[f32; 3]; 2],
    labels: [[f32; 3]; 2],
    // labels: ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
}

impl DIter {
    /// Construct a new instance from a DMatrixHandle created by the XGBoost C API.
    fn new(
        iter_handle: xgboost_bib::DataIterHandle,
        mut proxy_handle: xgboost_bib::DMatrixHandle,
    ) -> XGBResult<Self> {
        let proxi = unsafe { xgboost_bib::XGProxyDMatrixCreate(&mut proxy_handle) };

        let data = arr2(&[
            [1.0, 2.0, 3.0, 4.0, 5.0, 3.0],
            [1.0, 2.0, 3.0, 4.0, 5.0, 3.0],
            [1.0, 2.0, 3.0, 4.0, 5.0, 3.0],
            [1.0, 2.0, 3.0, 4.0, 5.0, 3.0],
            [1.0, 2.0, 3.0, 4.0, 5.0, 3.0],
            [1.0, 2.0, 3.0, 4.0, 5.0, 3.0],
        ]);

        let d = [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]];
        let l = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]];

        let labels = arr2(&[[16.0], [16.0], [16.0], [16.0], [16.0], [16.0]]);

        Ok(DIter {
            handle: iter_handle,
            n: 2,
            lengths: 2,
            cur_it: 0,
            proxy: proxy_handle,
            data: d,
            labels: l,
        })
    }

    #[no_mangle]
    pub extern "C" fn reset(handle: *mut ffi::c_void) {
        let data_iter: &mut DIter = unsafe { &mut *(handle as *mut DIter) };

        // println!("Setting cur_it to zero");
        // println!("current it: {:?}", data_iter.cur_it);

        data_iter.cur_it = 0;
    }

    #[no_mangle]
    extern "C" fn next(handle: *mut ffi::c_void) -> i32 {
        let data_iter: &mut DIter = unsafe { &mut *(handle as *mut DIter) };

        if data_iter.n == data_iter.cur_it {
            data_iter.cur_it = 0;
            return 0;
        } else {
            let data_ptr: *mut ffi::c_void =
                &mut data_iter.data[data_iter.cur_it] as *mut _ as *mut ffi::c_void;
            let lbl_ptr: *mut ffi::c_void =
                &mut data_iter.labels[data_iter.cur_it] as *mut _ as *mut ffi::c_void;

            let data_ptr_address = data_ptr as usize;

            let array_config = format!(
                "
                {{ \"data\": [{data_ptr_address},false], \"shape\": [3, 1], \"typestr\": \"<f4\", \"version\": 3}}
                "
            );

            let array_config_cstr = ffi::CString::new(array_config).unwrap();
            let field = ffi::CString::new(r#"label"#).unwrap();

            let dense_data = unsafe {
                xgboost_bib::XGProxyDMatrixSetDataDense(data_iter.proxy, array_config_cstr.as_ptr())
            };
            let dense_info = unsafe {
                xgboost_bib::XGDMatrixSetDenseInfo(
                    data_iter.proxy,
                    field.as_ptr(),
                    lbl_ptr,
                    3 as u64,
                    1,
                )
            };

            data_iter.cur_it += 1;
            return 1;
        }
    }
    fn free(&mut self) {
        drop(self);
    }
}

/// Data matrix used throughout XGBoost for training/predicting [`Booster`](struct.Booster.html) models.
///
/// It's used as a container for both features (i.e. a row for every instance), and an optional true label for that
/// instance (as an `f32` value).
///
/// Can be created files, or from dense or sparse
/// ([CSR](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format))
/// or [CSC](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_column_(CSC_or_CCS))) matrices.
///
/// # Examples
///
/// ## Load from file
///
/// Load matrix from file in [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) or binary format.
///
/// ```should_panic
/// use xgboost::DMatrix;
///
/// let dmat = DMatrix::load("somefile.txt").unwrap();
/// ```
///
/// ## Create from dense array
///
/// ```
/// use xgboost::DMatrix;
///
/// let data = &[1.0, 0.5, 0.2, 0.2,
///              0.7, 1.0, 0.1, 0.1,
///              0.2, 0.0, 0.0, 1.0];
/// let num_rows = 3;
/// let mut dmat = DMatrix::from_dense(data, num_rows).unwrap();
/// assert_eq!(dmat.shape(), (3, 4));
///
/// // set true labels for each row
/// dmat.set_labels(&[1.0, 0.0, 1.0]);
/// ```
///
/// ## Create from sparse CSR matrix
///
/// Create from sparse representation of
/// ```text
/// [[1.0, 0.0, 2.0],
///  [0.0, 0.0, 3.0],
///  [4.0, 5.0, 6.0]]
/// ```
///
/// ```
/// use xgboost::DMatrix;
///
/// let indptr = &[0, 2, 3, 6];
/// let indices = &[0, 2, 2, 0, 1, 2];
/// let data = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let dmat = DMatrix::from_csr(indptr, indices, data, None).unwrap();
/// assert_eq!(dmat.shape(), (3, 3));
/// ```
#[derive(Debug)]
pub struct DMatrix {
    pub(super) handle: xgboost_bib::DMatrixHandle,
    num_rows: usize,
    num_cols: usize,
}

impl DMatrix {
    /// Construct a new instance from a DMatrixHandle created by the XGBoost C API.
    fn new(handle: xgboost_bib::DMatrixHandle) -> XGBResult<Self> {
        // number of rows/cols are frequently read throughout applications, so more convenient to pull them out once
        // when the matrix is created, instead of having to check errors each time XGDMatrixNum* is called
        let mut out = 0;
        xgb_call!(xgboost_bib::XGDMatrixNumRow(handle, &mut out))?;
        let num_rows = out as usize;

        let mut out = 0;
        xgb_call!(xgboost_bib::XGDMatrixNumCol(handle, &mut out))?;
        let num_cols = out as usize;
        info!("Loaded DMatrix with shape: {}x{}", num_rows, num_cols);
        Ok(DMatrix {
            handle,
            num_rows,
            num_cols,
        })
    }

    /// Create a new `DMatrix` from dense array in row-major order.
    ///
    /// E.g. the matrix
    /// ```text
    /// [[1.0, 2.0],
    ///  [3.0, 4.0],
    ///  [5.0, 6.0]]
    /// ```
    /// would be represented converted into a `DMatrix` with
    /// ```
    /// use xgboost::DMatrix;
    ///
    /// let data = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// let num_rows = 3;
    /// let dmat = DMatrix::from_dense(data, num_rows).unwrap();
    /// ```
    pub fn from_dense(data: &[f32], num_rows: usize) -> XGBResult<Self> {
        let mut handle = ptr::null_mut();
        xgb_call!(xgboost_bib::XGDMatrixCreateFromMat(
            data.as_ptr(),
            num_rows as xgboost_bib::bst_ulong,
            (data.len() / num_rows) as xgboost_bib::bst_ulong,
            0.0, // TODO: can values be missing here?
            &mut handle
        ))?;
        Ok(DMatrix::new(handle)?)
    }

    pub fn from_callback(data: &[f32], num_rows: usize) -> XGBResult<Self> {
        let mut handle = ptr::null_mut();
        xgb_call!(xgboost_bib::XGDMatrixCreateFromMat(
            data.as_ptr(),
            num_rows as xgboost_bib::bst_ulong,
            (data.len() / num_rows) as xgboost_bib::bst_ulong,
            0.0, // TODO: can values be missing here?
            &mut handle
        ))?;
        Ok(DMatrix::new(handle)?)
    }

    /// Create a new `DMatrix` from a sparse
    /// [CSR](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)) matrix.
    ///
    /// Uses standard CSR representation where the column indices for row _i_ are stored in
    /// `indices[indptr[i]:indptr[i+1]]` and their corresponding values are stored in
    /// `data[indptr[i]:indptr[i+1]`.
    ///
    /// If `num_cols` is set to None, number of columns will be inferred from given data.
    pub fn from_csr(
        indptr: &[usize],
        indices: &[usize],
        data: &[f32],
        num_cols: Option<usize>,
    ) -> XGBResult<Self> {
        assert_eq!(indices.len(), data.len());
        let mut handle = ptr::null_mut();
        let indptr: Vec<u64> = indptr.iter().map(|x| *x as u64).collect();
        let indices: Vec<u32> = indices.iter().map(|x| *x as u32).collect();
        let num_cols = num_cols.unwrap_or(0); // infer from data if 0
        xgb_call!(xgboost_bib::XGDMatrixCreateFromCSREx(
            indptr.as_ptr(),
            indices.as_ptr(),
            data.as_ptr(),
            indptr.len().try_into().unwrap(),
            data.len().try_into().unwrap(),
            num_cols.try_into().unwrap(),
            &mut handle
        ))?;
        Ok(DMatrix::new(handle)?)
    }

    /// Create a new `DMatrix` from a sparse
    /// [CSC](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_column_(CSC_or_CCS))) matrix.
    ///
    /// Uses standard CSC representation where the row indices for column _i_ are stored in
    /// `indices[indptr[i]:indptr[i+1]]` and their corresponding values are stored in
    /// `data[indptr[i]:indptr[i+1]`.
    ///
    /// If `num_rows` is set to None, number of rows will be inferred from given data.
    pub fn from_csc(
        indptr: &[usize],
        indices: &[usize],
        data: &[f32],
        num_rows: Option<usize>,
    ) -> XGBResult<Self> {
        assert_eq!(indices.len(), data.len());
        let mut handle = ptr::null_mut();
        let indptr: Vec<u64> = indptr.iter().map(|x| *x as u64).collect();
        let indices: Vec<u32> = indices.iter().map(|x| *x as u32).collect();
        let num_rows = num_rows.unwrap_or(0); // infer from data if 0
        xgb_call!(xgboost_bib::XGDMatrixCreateFromCSCEx(
            indptr.as_ptr(),
            indices.as_ptr(),
            data.as_ptr(),
            indptr.len().try_into().unwrap(),
            data.len().try_into().unwrap(),
            num_rows.try_into().unwrap(),
            &mut handle
        ))?;
        Ok(DMatrix::new(handle)?)
    }

    pub fn from_iter() -> XGBResult<Self> {
        let iter_handle = ptr::null_mut();
        let proxy_handle = ptr::null_mut();
        let mut iter = DIter::new(iter_handle, proxy_handle).unwrap();

        let iter_ptr: *mut ffi::c_void = &mut iter as *mut _ as *mut ffi::c_void;

        let config = ffi::CString::new("{\"missing\": NaN, \"cache_prefix\": \"cache\"}").unwrap();
        let mut dm_handle: DMatrixHandle = ptr::null_mut();

        let from_callback = unsafe {
            xgboost_bib::XGDMatrixCreateFromCallback(
                iter_ptr,
                iter.proxy,
                Some(DIter::reset),
                Some(DIter::next),
                config.as_ptr(),
                &mut dm_handle,
            )
        };

        let dm = DMatrix::new(dm_handle).unwrap();
        Ok(dm)
    }

    /// Create a new `DMatrix` from given file.
    ///
    /// Supports text files in [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) format, CSV,
    /// binary files written either by `save`, or from another XGBoost library.
    ///
    /// For more details on accepted formats, seem the
    /// [XGBoost input format](https://xgboost.readthedocs.io/en/latest/tutorials/input_format.html)
    /// documentation.
    ///
    /// # LIBSVM format
    ///
    /// Specified data in a sparse format as:
    /// ```text
    /// <label> <index>:<value> [<index>:<value> ...]
    /// ```
    ///
    /// E.g.
    /// ```text
    /// 0 1:1 9:0 11:0
    /// 1 9:1 11:0.375 15:1
    /// 0 1:0 8:0.22 11:1
    /// ```
    pub fn load<P: AsRef<Path>>(path: P) -> XGBResult<Self> {
        let path_as_string = path.as_ref().display().to_string();
        let path_as_bytes = Path::new(&path_as_string).as_os_str().as_bytes();

        let mut handle = ptr::null_mut();
        let path_cstr = ffi::CString::new(path_as_bytes).unwrap();
        let silent = true;
        xgb_call!(xgboost_bib::XGDMatrixCreateFromFile(
            path_cstr.as_ptr(),
            silent as i32,
            &mut handle
        ))?;
        Ok(DMatrix::new(handle)?)
    }

    /// Serialise this `DMatrix` as a binary file to given path.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> XGBResult<()> {
        dbg!("Writing DMatrix to: {}", path.as_ref().display());
        let fname = ffi::CString::new(path.as_ref().as_os_str().as_bytes()).unwrap();
        let silent = true;
        xgb_call!(xgboost_bib::XGDMatrixSaveBinary(
            self.handle,
            fname.as_ptr(),
            silent as i32
        ))
    }

    /// Get the number of rows in this matrix.
    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    /// Get the number of columns in this matrix.
    pub fn num_cols(&self) -> usize {
        self.num_cols
    }

    /// Get the shape (rows x columns) of this matrix.
    pub fn shape(&self) -> (usize, usize) {
        (self.num_rows(), self.num_cols())
    }

    /// Get a new DMatrix as a containing only given indices.
    pub fn slice(&self, indices: &[usize]) -> XGBResult<DMatrix> {
        dbg!("Slicing {} rows from DMatrix", indices.len());
        let mut out_handle = ptr::null_mut();
        let indices: Vec<i32> = indices.iter().map(|x| *x as i32).collect();
        xgb_call!(xgboost_bib::XGDMatrixSliceDMatrix(
            self.handle,
            indices.as_ptr(),
            indices.len() as xgboost_bib::bst_ulong,
            &mut out_handle
        ))?;
        Ok(DMatrix::new(out_handle)?)
    }

    /// Get ground truth labels for each row of this matrix.
    pub fn get_labels(&self) -> XGBResult<&[f32]> {
        self.get_float_info(KEY_LABEL)
    }

    /// Set ground truth labels for each row of this matrix.
    pub fn set_labels(&mut self, array: &[f32]) -> XGBResult<()> {
        self.set_float_info(KEY_LABEL, array)
    }

    /// Get weights of each instance.
    pub fn get_weights(&self) -> XGBResult<&[f32]> {
        self.get_float_info(KEY_WEIGHT)
    }

    /// Set weights of each instance.
    pub fn set_weights(&mut self, array: &[f32]) -> XGBResult<()> {
        self.set_float_info(KEY_WEIGHT, array)
    }

    /// Get base margin.
    pub fn get_base_margin(&self) -> XGBResult<&[f32]> {
        self.get_float_info(KEY_BASE_MARGIN)
    }

    /// Set base margin.
    ///
    /// If specified, xgboost will start from this margin, can be used to specify initial prediction to boost from.
    pub fn set_base_margin(&mut self, array: &[f32]) -> XGBResult<()> {
        self.set_float_info(KEY_BASE_MARGIN, array)
    }

    /// Set the index for the beginning and end of a group.
    ///
    /// Needed when the learning task is ranking.
    ///
    /// See the XGBoost documentation for more information.
    pub fn set_group(&mut self, group: &[u32]) -> XGBResult<()> {
        // same as xgb_call!(xgboost_bib::XGDMatrixSetGroup(self.handle, group.as_ptr(), group.len() as u64))
        self.set_uint_info(KEY_GROUP, group)
    }

    /// Get the index for the beginning and end of a group.
    ///
    /// Needed when the learning task is ranking.
    ///
    /// See the XGBoost documentation for more information.
    pub fn get_group(&self) -> XGBResult<&[u32]> {
        self.get_uint_info(KEY_GROUP_PTR)
    }

    fn get_float_info(&self, field: &str) -> XGBResult<&[f32]> {
        let field = ffi::CString::new(field).unwrap();
        let mut out_len = 0;
        let mut out_dptr = ptr::null();
        xgb_call!(xgboost_bib::XGDMatrixGetFloatInfo(
            self.handle,
            field.as_ptr(),
            &mut out_len,
            &mut out_dptr
        ))?;

        Ok(unsafe { slice::from_raw_parts(out_dptr as *mut c_float, out_len as usize) })
    }

    fn set_float_info(&mut self, field: &str, array: &[f32]) -> XGBResult<()> {
        let field = ffi::CString::new(field).unwrap();
        xgb_call!(xgboost_bib::XGDMatrixSetFloatInfo(
            self.handle,
            field.as_ptr(),
            array.as_ptr(),
            array.len() as u64
        ))
    }

    fn get_uint_info(&self, field: &str) -> XGBResult<&[u32]> {
        let field = ffi::CString::new(field).unwrap();
        let mut out_len = 0;
        let mut out_dptr = ptr::null();
        xgb_call!(xgboost_bib::XGDMatrixGetUIntInfo(
            self.handle,
            field.as_ptr(),
            &mut out_len,
            &mut out_dptr
        ))?;
        Ok(unsafe { slice::from_raw_parts(out_dptr as *mut c_uint, out_len as usize) })
    }

    fn set_uint_info(&mut self, field: &str, array: &[u32]) -> XGBResult<()> {
        let field = ffi::CString::new(field).unwrap();
        xgb_call!(xgboost_bib::XGDMatrixSetUIntInfo(
            self.handle,
            field.as_ptr(),
            array.as_ptr(),
            array.len() as u64
        ))
    }
}

impl Drop for DMatrix {
    fn drop(&mut self) {
        xgb_call!(xgboost_bib::XGDMatrixFree(self.handle)).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        parameters::{self, learning, tree, BoosterParameters},
        Booster,
    };

    use super::*;
    use ndarray::arr2;
    use tempfile;
    use xgboost_bib::{XGBoosterEvalOneIter, XGBoosterUpdateOneIter};
    fn read_train_matrix() -> XGBResult<DMatrix> {
        println!("loading mat");
        DMatrix::load("data.csv?format=csv")
    }

    #[test]
    fn read_matrix() {
        assert!(read_train_matrix().is_ok());
    }

    #[test]
    fn read_num_rows() {
        assert_eq!(read_train_matrix().unwrap().num_rows(), 150);
    }

    #[test]
    fn read_num_cols() {
        assert_eq!(read_train_matrix().unwrap().num_cols(), 5);
    }

    #[test]
    fn writing_and_reading() {
        let dmat = read_train_matrix().unwrap();

        let tmp_dir = tempfile::tempdir().expect("failed to create temp dir");
        let out_path = tmp_dir.path().join("dmat.bin");
        dmat.save(&out_path).unwrap();

        let dmat2 = DMatrix::load(&out_path).unwrap();

        assert_eq!(dmat.num_rows(), dmat2.num_rows());
        assert_eq!(dmat.num_cols(), dmat2.num_cols());
        // TODO: check contents as well, if possible
    }

    #[test]
    fn get_set_labels() {
        let mut dmat = read_train_matrix().unwrap();
        assert_eq!(dmat.get_labels().unwrap().len(), 150);
        let labels = vec![0.0; dmat.get_labels().unwrap().len()];
        assert!(dmat.set_labels(&labels).is_ok());
        assert_eq!(dmat.get_labels().unwrap(), labels);
    }

    #[test]
    fn get_set_weights() {
        let mut dmat = read_train_matrix().unwrap();
        let empty_weights: Vec<f32> = vec![];
        assert_eq!(dmat.get_weights().unwrap(), empty_weights.as_slice());

        let weight = [1.0, 10.0, 44.9555];
        assert!(dmat.set_weights(&weight).is_ok());
        assert_eq!(dmat.get_weights().unwrap(), weight);
    }

    #[test]
    fn get_set_base_margin() {
        let mut dmat = read_train_matrix().unwrap();
        let empty_slice: Vec<f32> = vec![];
        assert_eq!(dmat.get_base_margin().unwrap(), empty_slice.as_slice());
        let base_margin = vec![1337.0; dmat.num_rows()];
        assert!(dmat.set_base_margin(&base_margin).is_ok());
        assert_eq!(dmat.get_base_margin().unwrap(), base_margin);
    }

    #[test]
    fn get_set_group() {
        let mut dmat = read_train_matrix().unwrap();
        let empty_slice: Vec<u32> = vec![];
        assert_eq!(dmat.get_group().unwrap(), empty_slice.as_slice());

        let group = [1];
        assert!(dmat.set_group(&group).is_ok());
        assert_eq!(dmat.get_group().unwrap(), &[0, 1]);
    }

    #[test]
    fn from_csr() {
        let indptr = [0, 2, 3, 6, 8];
        let indices = [0, 2, 2, 0, 1, 2, 1, 2];
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let dmat = DMatrix::from_csr(&indptr, &indices, &data, None).unwrap();
        assert_eq!(dmat.num_rows(), 4);
        assert_eq!(dmat.num_cols(), 0); // https://github.com/dmlc/xgboost/pull/7265

        let dmat = DMatrix::from_csr(&indptr, &indices, &data, Some(10)).unwrap();
        assert_eq!(dmat.num_rows(), 4);
        assert_eq!(dmat.num_cols(), 10);
    }

    #[test]
    fn from_csc() {
        let indptr = [0, 2, 3, 6, 8];
        let indices = [0, 2, 2, 0, 1, 2, 1, 2];
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let dmat = DMatrix::from_csc(&indptr, &indices, &data, None).unwrap();
        assert_eq!(dmat.num_rows(), 3);
        assert_eq!(dmat.num_cols(), 4);

        let dmat = DMatrix::from_csc(&indptr, &indices, &data, Some(10)).unwrap();
        assert_eq!(dmat.num_rows(), 10);
        assert_eq!(dmat.num_cols(), 4);
    }

    #[test]
    fn from_dense() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let num_rows = 2;

        let dmat = DMatrix::from_dense(&data, num_rows).unwrap();
        assert_eq!(dmat.num_rows(), 2);
        assert_eq!(dmat.num_cols(), 3);

        let data = vec![1.0, 2.0, 3.0];
        let num_rows = 3;

        let dmat = DMatrix::from_dense(&data, num_rows).unwrap();
        assert_eq!(dmat.num_rows(), 3);
        assert_eq!(dmat.num_cols(), 1);
    }

    #[test]
    fn slice_from_indices() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let num_rows = 4;

        let dmat = DMatrix::from_dense(&data, num_rows).unwrap();

        assert_eq!(dmat.shape(), (4, 2));

        assert_eq!(dmat.slice(&[]).unwrap().shape(), (0, 2));
        assert_eq!(dmat.slice(&[1]).unwrap().shape(), (1, 2));
        assert_eq!(dmat.slice(&[0, 1]).unwrap().shape(), (2, 2));
        assert_eq!(dmat.slice(&[3, 2, 1]).unwrap().shape(), (3, 2));
    }

    #[test]
    fn slice() {
        let data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let num_rows = 4;

        let dmat = DMatrix::from_dense(&data, num_rows).unwrap();
        assert_eq!(dmat.shape(), (4, 3));

        assert_eq!(dmat.slice(&[0, 1, 2, 3]).unwrap().shape(), (4, 3));
        assert_eq!(dmat.slice(&[0, 1]).unwrap().shape(), (2, 3));
        assert_eq!(dmat.slice(&[1, 0]).unwrap().shape(), (2, 3));
        assert_eq!(dmat.slice(&[0, 1, 2]).unwrap().shape(), (3, 3));
        assert_eq!(dmat.slice(&[3, 2, 1]).unwrap().shape(), (3, 3));
    }

    // ---------------------------------------------------------------------
    #[test]
    fn external_mem() {
        let X = arr2(&[
            [1.0, 5.0, 10.0],
            [1.0, 5.0, 10.0],
            [1.0, 5.0, 10.0],
            [1.0, 5.0, 10.0],
            [1.0, 5.0, 10.0],
            [1.0, 5.0, 10.0],
            [1.0, 5.0, 10.0],
            [1.0, 5.0, 10.0],
            [1.0, 5.0, 10.0],
            [1.0, 5.0, 10.0],
            [1.0, 5.0, 10.0],
            [1.0, 5.0, 10.0],
            [1.0, 5.0, 10.0],
            [1.0, 5.0, 10.0],
            [1.0, 5.0, 10.0],
            [1.0, 5.0, 10.0],
            [1.0, 5.0, 10.0],
            [1.0, 5.0, 10.0],
            [1.0, 5.0, 10.0],
            [1.0, 5.0, 10.0],
        ]);

        let y = arr2(&[
            [16.0],
            [16.0],
            [16.0],
            [16.0],
            [16.0],
            [16.0],
            [16.0],
            [16.0],
            [16.0],
            [16.0],
            [16.0],
            [16.0],
            [16.0],
            [16.0],
            [16.0],
            [16.0],
            [16.0],
            [16.0],
            [16.0],
            [16.0],
        ]);

        // let mut booster = Booster::new(&params).unwrap();
        let train_shape = X.raw_dim();
        let num_rows_train = train_shape[0];

        let mut dm = DMatrix::from_dense(
            X.into_shape(train_shape[0] * train_shape[1])
                .unwrap()
                .as_slice()
                .unwrap(),
            num_rows_train,
        )
        .unwrap();

        dm.set_labels(y.as_slice().unwrap()).unwrap();

        let mut booster =
            Booster::new_with_cached_dmats(&BoosterParameters::default(), &[&dm]).unwrap();

        let l = dm.get_labels().unwrap();
        println!("{:?}", l);
        for i in 0..2 {
            booster.update(&dm, i).unwrap();
            let eval = booster.evaluate(&dm).unwrap();
            println!("{:?}", eval);
        }

        booster.save(Path::new(&String::from("dm.json"))).unwrap();
        drop(booster);
    }
    fn train_model(xy: DMatrix) {
        let mut s = vec![xy.handle];
        let mut bst_handle = ptr::null_mut();
        let _booster_create =
            unsafe { xgboost_bib::XGBoosterCreate(s.as_ptr(), 1, &mut bst_handle) };

        let name_cstr = ffi::CString::new("tree_method").unwrap();
        let value_cstr = ffi::CString::new("approx").unwrap();

        let _booster_set_param = unsafe {
            xgboost_bib::XGBoosterSetParam(bst_handle, name_cstr.as_ptr(), value_cstr.as_ptr())
        };

        let name_cstr = ffi::CString::new("objective").unwrap();
        let value_cstr = ffi::CString::new("reg:squarederror").unwrap();

        let _booster_set_param = unsafe {
            xgboost_bib::XGBoosterSetParam(bst_handle, name_cstr.as_ptr(), value_cstr.as_ptr())
        };

        // TODO:
        // let mut evnames: Vec<ffi::CString> = Vec::with_capacity(names.len());
        let tr = ffi::CString::new("train").unwrap();
        let mut validation_names: Vec<*const libc::c_char> = vec![tr.as_ptr()];

        let mut validation_result = ptr::null();

        for i in 0..10 {
            unsafe {
                xgboost_bib::XGBoosterUpdateOneIter(bst_handle, i, xy.handle);
                xgboost_bib::XGBoosterEvalOneIter(
                    bst_handle,
                    i,
                    s.as_mut_ptr(),
                    validation_names.as_mut_ptr(),
                    1,
                    &mut validation_result,
                );
            }

            let out = unsafe {
                ffi::CStr::from_ptr(validation_result)
                    .to_str()
                    .unwrap()
                    .to_owned()
            };

            println!("{}", out);
        }
    }
    #[test]
    fn mainy() {
        let dm = DMatrix::from_iter().unwrap();
        train_model(dm);
    }
}
