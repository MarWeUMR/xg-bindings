use std::ptr;

use super::{XGBError, XGBResult};
use ndarray::arr1;
use ndarray::arr2;
use ndarray::ArrayBase;
use ndarray::Dim;
use ndarray::OwnedRepr;
use std::ffi;
use xgboost_bib::DMatrixHandle;

#[derive(Debug)]
#[repr(C)]
pub struct DIter {
    // pub(super) handle: xgboost_bib::DataIterHandle,
    n: usize,
    lengths: usize,
    cur_it: usize,
    proxy: DMatrixHandle,
    data: ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    labels: ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
}

impl DIter {
    /// Construct a new instance from a DMatrixHandle created by the XGBoost C API.
    // fn new(handle: xgboost_bib::DataIterHandle, handle2: &mut xgboost_bib::DMatrixHandle) -> XGBResult<Self> {
    fn new(handle2: &mut xgboost_bib::DMatrixHandle) -> XGBResult<Self> {
        let proxy_handle = handle2;
        let proxi = unsafe { xgboost_bib::XGProxyDMatrixCreate(proxy_handle as *mut *mut _) };
        println!("create proxy: {:?}", proxi);

        let data = arr2(&[
            [1.0, 2.0, 3.0, 4.0, 5.0, 3.0],
            [1.0, 2.0, 3.0, 4.0, 5.0, 3.0],
            [1.0, 2.0, 3.0, 4.0, 5.0, 3.0],
            [1.0, 2.0, 3.0, 4.0, 5.0, 3.0],
            [1.0, 2.0, 3.0, 4.0, 5.0, 3.0],
            [1.0, 2.0, 3.0, 4.0, 5.0, 3.0],
        ]);

        let labels = arr2(&[[16.0], [16.0], [16.0], [16.0], [16.0], [16.0]]);

        Ok(DIter {
            n: 3,
            lengths: 2,
            cur_it: 0,
            proxy: *proxy_handle,
            data,
            labels,
        })
    }

    // fn reset(&mut self) {
    //     self.cur_it = 0;
    // }

    #[no_mangle]
    pub extern "C" fn reset(handle: *mut ffi::c_void) {
        let data_iter: &mut DIter = unsafe { &mut *(handle as *mut DIter) };

        println!("Setting cur_it to zero");

        data_iter.cur_it = 0;
    }

    #[no_mangle]
    extern "C" fn next(handle: *mut ffi::c_void) -> i32 {
        let data_iter: &mut DIter = unsafe { &mut *(handle as *mut DIter) };

        if data_iter.n == data_iter.cur_it {
            return 0;
        } else {
            let array_config = ffi::CString::new(
                r#"{"data": [1,false], "shape": [2, 1], "typestr": "<f4", "version": 3}"#,
            )
            .unwrap();
            let field = ffi::CString::new(r#"label"#).unwrap();

            // let mut data = &[1.0, 2.0, 3.0, 4.0, 5.0, 3.0];
            let data_ptr: *mut ffi::c_void = &mut data_iter.data.row(0) as *mut _ as *mut ffi::c_void;

            let dense_data = unsafe {
                xgboost_bib::XGProxyDMatrixSetDataDense(data_iter.proxy, array_config.as_ptr())
            };
            let dense_info = unsafe {
                xgboost_bib::XGDMatrixSetDenseInfo(data_iter.proxy, field.as_ptr(), data_ptr, 2, 1)
            };

            println!("set dense data: {:?}", dense_data);
            println!("set dense info: {:?}", dense_info);

            return 1;
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn data_iter() {
        // let mut hd = ptr::null_mut();
        let mut hd2 = ptr::null_mut();
        let mut iter = DIter::new(&mut hd2).unwrap();

        let iter_ptr: *mut ffi::c_void = &mut iter as *mut _ as *mut ffi::c_void;
        let proxy_ptr: *mut ffi::c_void = &mut iter.proxy as *mut _ as *mut ffi::c_void;

        let config = ffi::CString::new("{\"missing\": NaN, \"cache_prefix\": \"cache\"}").unwrap();
        let mut dm_handle = ptr::null_mut();

        // DIter::re(di_ptr);

        let iii = unsafe {
            xgboost_bib::XGDMatrixCreateFromCallback(
                iter_ptr,
                iter.proxy,
                Some(DIter::reset),
                Some(DIter::next),
                config.as_ptr(),
                &mut dm_handle,
            )
        };

        println!("create from callback: {:?}", iii);
    }
}
