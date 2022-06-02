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
        // println!("proxy pointer in init (before) -> {:p}", proxy_handle);
        // fn new(handle2: &mut xgboost_bib::DMatrixHandle) -> XGBResult<Self> {
        // let proxy_handle = ptr::null_mut();
        let proxi = unsafe { xgboost_bib::XGProxyDMatrixCreate(&mut proxy_handle) };
        // println!("create proxy: {:?}", proxi);

        // println!("proxy pointer in init (after) -> {:p}", proxy_handle);
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

            // println!("proxy pointer after next -> {:p}", data_iter.proxy);
            // println!("set dense data: {:?}", dense_data);
            // println!("set dense info: {:?}", dense_info);

            data_iter.cur_it += 1;
            return 1;
        }
    }
    fn free(&mut self) {
        drop(self);
    }
}

#[cfg(test)]
mod tests {

    use crate::DMatrix;

    use super::*;
    use xgboost_bib::*;

    fn train_model(xy: DMatrix) {
        let s = vec![xy.handle];
        let mut bst_handle = ptr::null_mut();
        let booster_create =
            unsafe { xgboost_bib::XGBoosterCreate(s.as_ptr(), 1, &mut bst_handle) };

        let name_cstr = ffi::CString::new("tree_method").unwrap();
        let value_cstr = ffi::CString::new("approx").unwrap();

        let booster_set_param = unsafe {
            xgboost_bib::XGBoosterSetParam(bst_handle, name_cstr.as_ptr(), value_cstr.as_ptr())
        };

        let name_cstr = ffi::CString::new("objective").unwrap();
        let value_cstr = ffi::CString::new("reg:squarederror").unwrap();

        
        let  booster_set_param = unsafe {
            xgboost_bib::XGBoosterSetParam(bst_handle, name_cstr.as_ptr(), value_cstr.as_ptr())
        };
    }

    #[test]
    fn data_iter() {
        let iter_handle = ptr::null_mut();
        let proxy_handle = ptr::null_mut();
        // println!("proxy pointer pre init -> {:p}", proxy_handle);
        let mut iter = DIter::new(iter_handle, proxy_handle).unwrap();

        // println!("proxy pointer post init -> {:p}", proxy_handle);
        // println!("proxy pointer post init from struct -> {:p}", iter.proxy);

        let iter_ptr: *mut ffi::c_void = &mut iter as *mut _ as *mut ffi::c_void;
        // let proxy_ptr: *mut ffi::c_void = &mut iter.proxy as *mut _ as *mut ffi::c_void;

        let config = ffi::CString::new("{\"missing\": NaN, \"cache_prefix\": \"cache\"}").unwrap();
        let mut dm_handle: DMatrixHandle = ptr::null_mut();

        // println!("proxy pointer 3 -> {:p}", proxy_ptr);

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
