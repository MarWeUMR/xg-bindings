use crate::dmatrix::DMatrix;
use crate::error::XGBError;
use indexmap::IndexMap;
use libc;
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::{self, BufRead, BufReader, Write};
use std::iter::zip;
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::{ffi, fmt, fs::File, ptr, slice};
use tempfile;
use xgboost_bib;

use super::XGBResult;
use crate::parameters::{BoosterParameters, TrainingParameters};

pub type CustomObjective = fn(&[f32], &DMatrix) -> (Vec<f32>, Vec<f32>);

/// Used to control the return type of predictions made by C Booster API.
enum PredictOption {
    OutputMargin,
    PredictLeaf,
    PredictContribitions,
    //ApproximateContributions,
    PredictInteractions,
}

impl PredictOption {
    /// Convert list of options into a bit mask.
    fn options_as_mask(options: &[PredictOption]) -> i32 {
        let mut option_mask = 0x00;
        for option in options {
            let value = match *option {
                PredictOption::OutputMargin => 0x01,
                PredictOption::PredictLeaf => 0x02,
                PredictOption::PredictContribitions => 0x04,
                //PredictOption::ApproximateContributions => 0x08,
                PredictOption::PredictInteractions => 0x10,
            };
            option_mask |= value;
        }

        option_mask
    }
}

/// Core model in XGBoost, containing functions for training, evaluating and predicting.
///
/// Usually created through the [`train`](struct.Booster.html#method.train) function, which
/// creates and trains a Booster in a single call.
///
/// For more fine grained usage, can be created using [`new`](struct.Booster.html#method.new) or
/// [`new_with_cached_dmats`](struct.Booster.html#method.new_with_cached_dmats), then trained by calling
/// [`update`](struct.Booster.html#method.update) or [`update_custom`](struct.Booster.html#method.update_custom)
/// in a loop.
pub struct Booster {
    handle: xgboost_bib::BoosterHandle,
}

unsafe impl Send for Booster {}
unsafe impl Sync for Booster {}

impl Booster {
    /// Create a new Booster model with given parameters.
    ///
    /// This model can then be trained using calls to update/boost as appropriate.
    ///
    /// The [`train`](struct.Booster.html#method.train)  function is often a more convenient way of constructing,
    /// training and evaluating a Booster in a single call.
    pub fn new(params: &BoosterParameters) -> XGBResult<Self> {
        Self::new_with_cached_dmats(params, &[])
    }

    pub fn new_with_json_config(
        dmats: &[&DMatrix],

        config: HashMap<&str, &str>,
    ) -> XGBResult<Self> {
        let mut handle = ptr::null_mut();
        // TODO: check this is safe if any dmats are freed
        let s: Vec<xgboost_bib::DMatrixHandle> = dmats.iter().map(|x| x.handle).collect();
        xgb_call!(xgboost_bib::XGBoosterCreate(
            s.as_ptr(),
            dmats.len() as u64,
            &mut handle
        ))?;

        let mut booster = Booster { handle };
        booster.set_param_from_json(config);
        Ok(booster)
    }

    /// Create a new Booster model with given parameters and list of DMatrix to cache.
    ///
    /// Cached DMatrix can sometimes be used internally by XGBoost to speed up certain operations.
    pub fn new_with_cached_dmats(
        params: &BoosterParameters,
        dmats: &[&DMatrix],
    ) -> XGBResult<Self> {
        let mut handle = ptr::null_mut();
        // TODO: check this is safe if any dmats are freed
        let s: Vec<xgboost_bib::DMatrixHandle> = dmats.iter().map(|x| x.handle).collect();
        xgb_call!(xgboost_bib::XGBoosterCreate(
            s.as_ptr(),
            dmats.len() as u64,
            &mut handle
        ))?;

        let mut booster = Booster { handle };
        booster.set_params(params)?;
        Ok(booster)
    }

    /// Save this Booster as a binary file at given path.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> XGBResult<()> {
        debug!("Writing Booster to: {}", path.as_ref().display());
        let fname = ffi::CString::new(path.as_ref().as_os_str().as_bytes()).unwrap();
        xgb_call!(xgboost_bib::XGBoosterSaveModel(self.handle, fname.as_ptr()))
    }

    /// Load a Booster from a binary file at given path.
    pub fn load<P: AsRef<Path>>(path: P) -> XGBResult<Self> {
        debug!("Loading Booster from: {}", path.as_ref().display());

        // gives more control over error messages, avoids stack trace dump from C++
        if !path.as_ref().exists() {
            return Err(XGBError::new(format!(
                "File not found: {}",
                path.as_ref().display()
            )));
        }

        let fname = ffi::CString::new(path.as_ref().as_os_str().as_bytes()).unwrap();
        let mut handle = ptr::null_mut();
        xgb_call!(xgboost_bib::XGBoosterCreate(ptr::null(), 0, &mut handle))?;
        xgb_call!(xgboost_bib::XGBoosterLoadModel(handle, fname.as_ptr()))?;
        Ok(Booster { handle })
    }

    /// Load a Booster directly from a buffer.
    pub fn load_buffer(bytes: &[u8]) -> XGBResult<Self> {
        debug!("Loading Booster from buffer (length = {})", bytes.len());

        let mut handle = ptr::null_mut();
        xgb_call!(xgboost_bib::XGBoosterCreate(ptr::null(), 0, &mut handle))?;
        xgb_call!(xgboost_bib::XGBoosterLoadModelFromBuffer(
            handle,
            bytes.as_ptr() as *const _,
            bytes.len() as u64
        ))?;
        Ok(Booster { handle })
    }

    pub fn train_increment(params: &TrainingParameters, model_name: &str) -> XGBResult<Self> {
        let cached_dmats = {
            let mut dmats = vec![params.dtrain];
            if let Some(eval_sets) = params.evaluation_sets {
                for (dmat, _) in eval_sets {
                    dmats.push(*dmat);
                }
            }
            dmats
        };

        let path = Path::new(model_name);
        let bytes = std::fs::read(&path).expect("read saved booster file");
        let mut bst = Booster::load_buffer(&bytes[..]).expect("load booster from buffer");
        // let mut bst = Booster::new_with_cached_dmats(&params.booster_params, &cached_dmats)?;
        //let num_parallel_tree = 1;

        // load distributed code checkpoint from rabit
        let version = bst.load_rabit_checkpoint()?;
        debug!("Loaded Rabit checkpoint: version={}", version);
        assert!(unsafe { xgboost_bib::RabitGetWorldSize() != 1 || version == 0 });

        let _rank = unsafe { xgboost_bib::RabitGetRank() };
        let start_iteration = version / 2;
        //let mut nboost = start_iteration;

        for i in start_iteration..params.boost_rounds as i32 {
            // distributed code: need to resume to this point
            // skip first update if a recovery step
            if version % 2 == 0 {
                if let Some(objective_fn) = params.custom_objective_fn {
                    debug!("Boosting in round: {}", i);
                    bst.update_custom(params.dtrain, objective_fn)?;
                } else {
                    debug!("Updating in round: {}", i);
                    bst.update(params.dtrain, i)?;
                }
                bst.save_rabit_checkpoint()?;
            }

            assert!(unsafe {
                xgboost_bib::RabitGetWorldSize() == 1
                    || version == xgboost_bib::RabitVersionNumber()
            });

            //nboost += 1;

            if let Some(eval_sets) = params.evaluation_sets {
                let mut dmat_eval_results = bst.eval_set(eval_sets, i)?;

                if let Some(eval_fn) = params.custom_evaluation_fn {
                    let eval_name = "custom";
                    for (dmat, dmat_name) in eval_sets {
                        let margin = bst.predict_margin(dmat)?;
                        let eval_result = eval_fn(&margin, dmat);
                        let eval_results = dmat_eval_results
                            .entry(eval_name.to_string())
                            .or_insert_with(IndexMap::new);
                        eval_results.insert(dmat_name.to_string(), eval_result);
                    }
                }

                // convert to map of eval_name -> (dmat_name -> score)
                let mut eval_dmat_results = BTreeMap::new();
                for (dmat_name, eval_results) in &dmat_eval_results {
                    for (eval_name, result) in eval_results {
                        let dmat_results = eval_dmat_results
                            .entry(eval_name)
                            .or_insert_with(BTreeMap::new);
                        dmat_results.insert(dmat_name, result);
                    }
                }

                print!("[{}]", i);
                for (eval_name, dmat_results) in eval_dmat_results {
                    for (dmat_name, result) in dmat_results {
                        print!("\t{}-{}:{}", dmat_name, eval_name, result);
                    }
                }
                println!();
            }
        }

        Ok(bst)
    }

    pub fn train(
        evaluation_sets: Option<&[(&DMatrix, &str)]>,
        dtrain: &DMatrix,
        config: HashMap<&str, &str>,
        bst: Option<Booster>,
    ) -> XGBResult<Self> {
        let cached_dmats = {
            let mut dmats = vec![dtrain];
            if let Some(eval_sets) = evaluation_sets {
                for (dmat, _) in eval_sets {
                    dmats.push(*dmat);
                }
            }
            dmats
        };

        let mut bst: Booster = {
            if let Some(booster) = bst {
                let mut length: u64 = 0;
                let mut buffer_string = ptr::null();

                let _ = xgb_call!(xgboost_bib::XGBoosterSerializeToBuffer(
                    booster.handle,
                    &mut length,
                    &mut buffer_string
                ));

                let mut bst_handle = ptr::null_mut();

                let cached_dmat_handles: Vec<xgboost_bib::DMatrixHandle> =
                    cached_dmats.iter().map(|x| x.handle).collect();

                xgb_call!(xgboost_bib::XGBoosterCreate(
                    cached_dmat_handles.as_ptr(),
                    cached_dmats.len() as u64,
                    &mut bst_handle
                ))?;

                let mut bst_unserialize = Booster { handle: bst_handle };

                let _ = xgb_call!(xgboost_bib::XGBoosterUnserializeFromBuffer(
                    bst_unserialize.handle,
                    buffer_string as *mut ffi::c_void,
                    length,
                ));

                bst_unserialize.set_param_from_json(config);
                bst_unserialize
            } else {
                let bst = Booster::new_with_json_config(&cached_dmats, config)?;
                bst
            }
        };

        for i in 0..16 {
            bst.update(dtrain, i)?;

            if let Some(eval_sets) = evaluation_sets {
                let dmat_eval_results = bst.eval_set(eval_sets, i)?;

                // convert to map of eval_name -> (dmat_name -> score)
                let mut eval_dmat_results = BTreeMap::new();
                for (dmat_name, eval_results) in &dmat_eval_results {
                    for (eval_name, result) in eval_results {
                        let dmat_results = eval_dmat_results
                            .entry(eval_name)
                            .or_insert_with(BTreeMap::new);
                        dmat_results.insert(dmat_name, result);
                    }
                }

                print!("[{}]", i);
                for (eval_name, dmat_results) in eval_dmat_results {
                    for (dmat_name, result) in dmat_results {
                        print!("\t{}-{}:{}", dmat_name, eval_name, result);
                    }
                }
                println!();
            }
        }

        Ok(bst)
    }

    pub fn save_config(&self) -> String {
        /*
        json_string = ctypes.c_char_p()
        length = c_bst_ulong()
        _check_call(_LIB.XGBoosterSaveJsonConfig(
            self.handle,
            ctypes.byref(length),
            ctypes.byref(json_string)))
        assert json_string.value is not None
        result = json_string.value.decode()  # pylint: disable=no-member
        return result
        */

        // let json_string: libc::c_char =
        let mut length: u64 = 1;
        let mut json_string = ptr::null();

        let json = unsafe {
            xgboost_bib::XGBoosterSaveJsonConfig(self.handle, &mut length, &mut json_string)
        };

        let out = unsafe {
            ffi::CStr::from_ptr(json_string)
                .to_str()
                .unwrap()
                .to_owned()
        };

        println!("{}", json);
        println!("{}", out.clone());
        out
    }

    /// Update this Booster's parameters.
    pub fn set_params(&mut self, p: &BoosterParameters) -> XGBResult<()> {
        for (key, value) in p.as_string_pairs() {
            println!("challis: Setting parameter: {}={}", &key, &value);
            self.set_param(&key, &value)?;
        }
        Ok(())
    }

    /// Update this model by training it for one round with given training matrix.
    ///
    /// Uses XGBoost's objective function that was specificed in this Booster's learning objective parameters.
    ///
    /// * `dtrain` - matrix to train the model with for a single iteration
    /// * `iteration` - current iteration number
    pub fn update(&mut self, dtrain: &DMatrix, iteration: i32) -> XGBResult<()> {
        xgb_call!(xgboost_bib::XGBoosterUpdateOneIter(
            self.handle,
            iteration,
            dtrain.handle
        ))
    }

    /// Update this model by training it for one round with a custom objective function.
    pub fn update_custom(
        &mut self,
        dtrain: &DMatrix,
        objective_fn: CustomObjective,
    ) -> XGBResult<()> {
        let pred = self.predict(dtrain)?;
        let (gradient, hessian) = objective_fn(&pred.to_vec(), dtrain);
        self.boost(dtrain, &gradient, &hessian)
    }

    /// Update this model by directly specifying the first and second order gradients.
    ///
    /// This is typically used instead of `update` when using a customised loss function.
    ///
    /// * `dtrain` - matrix to train the model with for a single iteration
    /// * `gradient` - first order gradient
    /// * `hessian` - second order gradient
    fn boost(&mut self, dtrain: &DMatrix, gradient: &[f32], hessian: &[f32]) -> XGBResult<()> {
        if gradient.len() != hessian.len() {
            let msg = format!(
                "Mismatch between length of gradient and hessian arrays ({} != {})",
                gradient.len(),
                hessian.len()
            );
            return Err(XGBError::new(msg));
        }
        assert_eq!(gradient.len(), hessian.len());

        // TODO: _validate_feature_names
        let mut grad_vec = gradient.to_vec();
        let mut hess_vec = hessian.to_vec();
        xgb_call!(xgboost_bib::XGBoosterBoostOneIter(
            self.handle,
            dtrain.handle,
            grad_vec.as_mut_ptr(),
            hess_vec.as_mut_ptr(),
            grad_vec.len() as u64
        ))
    }

    fn eval_set(
        &self,
        evals: &[(&DMatrix, &str)],
        iteration: i32,
    ) -> XGBResult<IndexMap<String, IndexMap<String, f32>>> {
        let (dmats, names) = {
            let mut dmats = Vec::with_capacity(evals.len());
            let mut names = Vec::with_capacity(evals.len());
            for (dmat, name) in evals {
                dmats.push(dmat);
                names.push(*name);
            }
            (dmats, names)
        };
        assert_eq!(dmats.len(), names.len());

        let mut s: Vec<xgboost_bib::DMatrixHandle> = dmats.iter().map(|x| x.handle).collect();

        // build separate arrays of C strings and pointers to them to ensure they live long enough
        let mut evnames: Vec<ffi::CString> = Vec::with_capacity(names.len());
        let mut evptrs: Vec<*const libc::c_char> = Vec::with_capacity(names.len());

        for name in &names {
            let cstr = ffi::CString::new(*name).unwrap();
            evptrs.push(cstr.as_ptr());
            evnames.push(cstr);
        }

        // shouldn't be necessary, but guards against incorrect array sizing
        evptrs.shrink_to_fit();

        let mut out_result = ptr::null();
        xgb_call!(xgboost_bib::XGBoosterEvalOneIter(
            self.handle,
            iteration,
            s.as_mut_ptr(),
            evptrs.as_mut_ptr(),
            dmats.len() as u64,
            &mut out_result
        ))?;
        let out = unsafe { ffi::CStr::from_ptr(out_result).to_str().unwrap().to_owned() };
        Ok(Booster::parse_eval_string(&out, &names))
    }

    /// Evaluate given matrix against this model using metrics defined in this model's parameters.
    ///
    /// See parameter::learning::EvaluationMetric for a full list.
    ///
    /// Returns a map of evaluation metric name to score.
    pub fn evaluate(&self, dmat: &DMatrix) -> XGBResult<HashMap<String, f32>> {
        let name = "default";
        let mut eval = self.eval_set(&[(dmat, name)], 0)?;
        let mut result = HashMap::new();
        eval.remove(name).unwrap().into_iter().for_each(|(k, v)| {
            result.insert(k.to_owned(), v);
        });

        Ok(result)
    }

    /// Get a string attribute that was previously set for this model.
    pub fn get_attribute(&self, key: &str) -> XGBResult<Option<String>> {
        let key = ffi::CString::new(key).unwrap();
        let mut out_buf = ptr::null();
        let mut success = 0;
        xgb_call!(xgboost_bib::XGBoosterGetAttr(
            self.handle,
            key.as_ptr(),
            &mut out_buf,
            &mut success
        ))?;
        if success == 0 {
            return Ok(None);
        }
        assert!(success == 1);

        let c_str: &ffi::CStr = unsafe { ffi::CStr::from_ptr(out_buf) };
        let out = c_str.to_str().unwrap();
        Ok(Some(out.to_owned()))
    }

    /// Store a string attribute in this model with given key.
    pub fn set_attribute(&mut self, key: &str, value: &str) -> XGBResult<()> {
        let key = ffi::CString::new(key).unwrap();
        let value = ffi::CString::new(value).unwrap();
        xgb_call!(xgboost_bib::XGBoosterSetAttr(
            self.handle,
            key.as_ptr(),
            value.as_ptr()
        ))
    }

    /// Get names of all attributes stored in this model. Values can then be fetched with calls to `get_attribute`.
    pub fn get_attribute_names(&self) -> XGBResult<Vec<String>> {
        let mut out_len = 0;
        let mut out = ptr::null_mut();
        xgb_call!(xgboost_bib::XGBoosterGetAttrNames(
            self.handle,
            &mut out_len,
            &mut out
        ))?;

        let out_ptr_slice = unsafe { slice::from_raw_parts(out, out_len as usize) };
        let out_vec = out_ptr_slice
            .iter()
            .map(|str_ptr| unsafe { ffi::CStr::from_ptr(*str_ptr).to_str().unwrap().to_owned() })
            .collect();
        Ok(out_vec)
    }

    pub fn predict_from_dmat(
        &self,
        dmat: &DMatrix,
        out_shape: &[u64; 2],
        out_dim: &mut u64,
    ) -> XGBResult<Vec<f32>> {
        let json_config = format!("{{\"type\": 0,\"training\": false,\"iteration_begin\": 0,\"iteration_end\": 0,\"strict_shape\": true}}");

        let mut out_result = ptr::null();

        let c_json_config = ffi::CString::new(json_config).unwrap();

        xgb_call!(xgboost_bib::XGBoosterPredictFromDMatrix(
            self.handle,
            dmat.handle,
            c_json_config.as_ptr(),
            &mut out_shape.as_ptr(),
            out_dim,
            &mut out_result
        ))?;

        let out_len = out_shape[0];

        assert!(!out_result.is_null());
        let data = unsafe { slice::from_raw_parts(out_result, out_len as usize).to_vec() };
        Ok(data)
    }

    /// Predict results for given data.
    ///
    /// Returns an array containing one entry per row in the given data.
    pub fn predict(&self, dmat: &DMatrix) -> XGBResult<Vec<f32>> {
        let option_mask = PredictOption::options_as_mask(&[]);
        let ntree_limit = 0;
        let mut out_len = 0;
        let mut out_result = ptr::null();
        xgb_call!(xgboost_bib::XGBoosterPredict(
            self.handle,
            dmat.handle,
            option_mask,
            ntree_limit,
            0,
            &mut out_len,
            &mut out_result
        ))?;

        assert!(!out_result.is_null());
        let data = unsafe { slice::from_raw_parts(out_result, out_len as usize).to_vec() };
        Ok(data)
    }

    /// Predict margin for given data.
    ///
    /// Returns an array containing one entry per row in the given data.
    pub fn predict_margin(&self, dmat: &DMatrix) -> XGBResult<Vec<f32>> {
        let option_mask = PredictOption::options_as_mask(&[PredictOption::OutputMargin]);
        let ntree_limit = 0;
        let mut out_len = 0;
        let mut out_result = ptr::null();
        xgb_call!(xgboost_bib::XGBoosterPredict(
            self.handle,
            dmat.handle,
            option_mask,
            ntree_limit,
            1,
            &mut out_len,
            &mut out_result
        ))?;
        assert!(!out_result.is_null());
        let data = unsafe { slice::from_raw_parts(out_result, out_len as usize).to_vec() };
        Ok(data)
    }

    /// Get predicted leaf index for each sample in given data.
    ///
    /// Returns an array of shape (number of samples, number of trees) as tuple of (data, num_rows).
    ///
    /// Note: the leaf index of a tree is unique per tree, so e.g. leaf 1 could be found in both tree 1 and tree 0.
    pub fn predict_leaf(&self, dmat: &DMatrix) -> XGBResult<(Vec<f32>, (usize, usize))> {
        let option_mask = PredictOption::options_as_mask(&[PredictOption::PredictLeaf]);
        let ntree_limit = 0;
        let mut out_len = 0;
        let mut out_result = ptr::null();
        xgb_call!(xgboost_bib::XGBoosterPredict(
            self.handle,
            dmat.handle,
            option_mask,
            ntree_limit,
            0,
            &mut out_len,
            &mut out_result
        ))?;
        assert!(!out_result.is_null());

        let data = unsafe { slice::from_raw_parts(out_result, out_len as usize).to_vec() };
        let num_rows = dmat.num_rows();
        let num_cols = data.len() / num_rows;
        Ok((data, (num_rows, num_cols)))
    }

    /// Get feature contributions (SHAP values) for each prediction.
    ///
    /// The sum of all feature contributions is equal to the run untransformed margin value of the
    /// prediction.
    ///
    /// Returns an array of shape (number of samples, number of features + 1) as a tuple of
    /// (data, num_rows). The final column contains the bias term.
    pub fn predict_contributions(&self, dmat: &DMatrix) -> XGBResult<(Vec<f32>, (usize, usize))> {
        let option_mask = PredictOption::options_as_mask(&[PredictOption::PredictContribitions]);
        let ntree_limit = 0;
        let mut out_len = 0;
        let mut out_result = ptr::null();
        xgb_call!(xgboost_bib::XGBoosterPredict(
            self.handle,
            dmat.handle,
            option_mask,
            ntree_limit,
            0,
            &mut out_len,
            &mut out_result
        ))?;
        assert!(!out_result.is_null());

        let data = unsafe { slice::from_raw_parts(out_result, out_len as usize).to_vec() };
        let num_rows = dmat.num_rows();
        let num_cols = data.len() / num_rows;
        Ok((data, (num_rows, num_cols)))
    }

    /// Get SHAP interaction values for each pair of features for each prediction.
    ///
    /// The sum of each row (or column) of the interaction values equals the corresponding SHAP
    /// value (from `predict_contributions`), and the sum of the entire matrix equals the raw
    /// untransformed margin value of the prediction.
    ///
    /// Returns an array of shape (number of samples, number of features + 1, number of features + 1).
    /// The final row and column contain the bias terms.
    pub fn predict_interactions(
        &self,
        dmat: &DMatrix,
    ) -> XGBResult<(Vec<f32>, (usize, usize, usize))> {
        let option_mask = PredictOption::options_as_mask(&[PredictOption::PredictInteractions]);
        let ntree_limit = 0;
        let mut out_len = 0;
        let mut out_result = ptr::null();
        xgb_call!(xgboost_bib::XGBoosterPredict(
            self.handle,
            dmat.handle,
            option_mask,
            ntree_limit,
            0,
            &mut out_len,
            &mut out_result
        ))?;
        assert!(!out_result.is_null());

        let data = unsafe { slice::from_raw_parts(out_result, out_len as usize).to_vec() };
        let num_rows = dmat.num_rows();

        let dim = ((data.len() / num_rows) as f64).sqrt() as usize;
        Ok((data, (num_rows, dim, dim)))
    }

    /// Get a dump of this model as a string.
    ///
    /// * `with_statistics` - whether to include statistics in output dump
    /// * `feature_map` - if given, map feature IDs to feature names from given map
    pub fn dump_model(
        &self,
        with_statistics: bool,
        feature_map: Option<&FeatureMap>,
    ) -> XGBResult<String> {
        if let Some(fmap) = feature_map {
            let tmp_dir = match tempfile::tempdir() {
                Ok(dir) => dir,
                Err(err) => return Err(XGBError::new(err.to_string())),
            };

            let file_path = tmp_dir.path().join("fmap.txt");
            let mut file: File = match File::create(&file_path) {
                Ok(f) => f,
                Err(err) => return Err(XGBError::new(err.to_string())),
            };

            for (feature_num, (feature_name, feature_type)) in &fmap.0 {
                writeln!(file, "{}\t{}\t{}", feature_num, feature_name, feature_type).unwrap();
            }

            self.dump_model_fmap(with_statistics, Some(&file_path))
        } else {
            self.dump_model_fmap(with_statistics, None)
        }
    }

    fn dump_model_fmap(
        &self,
        with_statistics: bool,
        feature_map_path: Option<&PathBuf>,
    ) -> XGBResult<String> {
        let fmap = if let Some(path) = feature_map_path {
            ffi::CString::new(path.as_os_str().as_bytes()).unwrap()
        } else {
            ffi::CString::new("").unwrap()
        };
        let format = ffi::CString::new("text").unwrap();
        let mut out_len = 0;
        let mut out_dump_array = ptr::null_mut();
        xgb_call!(xgboost_bib::XGBoosterDumpModelEx(
            self.handle,
            fmap.as_ptr(),
            with_statistics as i32,
            format.as_ptr(),
            &mut out_len,
            &mut out_dump_array
        ))?;

        let out_ptr_slice = unsafe { slice::from_raw_parts(out_dump_array, out_len as usize) };
        let out_vec: Vec<String> = out_ptr_slice
            .iter()
            .map(|str_ptr| unsafe { ffi::CStr::from_ptr(*str_ptr).to_str().unwrap().to_owned() })
            .collect();

        assert_eq!(out_len as usize, out_vec.len());
        Ok(out_vec.join("\n"))
    }

    pub(crate) fn load_rabit_checkpoint(&self) -> XGBResult<i32> {
        let mut version = 0;
        xgb_call!(xgboost_bib::XGBoosterLoadRabitCheckpoint(
            self.handle,
            &mut version
        ))?;
        Ok(version)
    }

    pub(crate) fn save_rabit_checkpoint(&self) -> XGBResult<()> {
        xgb_call!(xgboost_bib::XGBoosterSaveRabitCheckpoint(self.handle))
    }

    fn set_param_from_json(&mut self, config: HashMap<&str, &str>) {
        for (k, v) in config.into_iter() {
            let name = ffi::CString::new(k).unwrap();
            let value = ffi::CString::new(v).unwrap();

            let setting_ok = unsafe {
                xgboost_bib::XGBoosterSetParam(self.handle, name.as_ptr(), value.as_ptr())
            };
        }

        // for (k, v) in zip(keys, values) {
        //     let name = ffi::CString::new(k).unwrap();
        //     let value = ffi::CString::new(v).unwrap();
        //
        //     let setting_ok = unsafe {
        //         xgboost_bib::XGBoosterSetParam(self.handle, name.as_ptr(), value.as_ptr())
        //     };
        // }
    }

    fn set_param(&mut self, name: &str, value: &str) -> XGBResult<()> {
        let name = ffi::CString::new(name).unwrap();
        let value = ffi::CString::new(value).unwrap();
        xgb_call!(xgboost_bib::XGBoosterSetParam(
            self.handle,
            name.as_ptr(),
            value.as_ptr()
        ))
    }

    fn parse_eval_string(eval: &str, evnames: &[&str]) -> IndexMap<String, IndexMap<String, f32>> {
        let mut result: IndexMap<String, IndexMap<String, f32>> = IndexMap::new();

        debug!("Parsing evaluation line: {}", &eval);
        for part in eval.split('\t').skip(1) {
            for evname in evnames {
                if part.starts_with(evname) {
                    let metric_parts: Vec<&str> =
                        part[evname.len() + 1..].split(':').into_iter().collect();
                    assert_eq!(metric_parts.len(), 2);
                    let metric = metric_parts[0];
                    let score = metric_parts[1].parse::<f32>().unwrap_or_else(|_| {
                        panic!("Unable to parse XGBoost metrics output: {}", eval)
                    });

                    let metric_map = result
                        .entry(evname.to_string())
                        .or_insert_with(IndexMap::new);
                    metric_map.insert(metric.to_owned(), score);
                }
            }
        }

        debug!("result: {:?}", &result);
        result
    }
}

impl Drop for Booster {
    fn drop(&mut self) {
        xgb_call!(xgboost_bib::XGBoosterFree(self.handle)).unwrap();
    }
}

/// Maps a feature index to a name and type, used when dumping models as text.
///
/// See [dump_model](struct.Booster.html#method.dump_model) for usage.
pub struct FeatureMap(BTreeMap<u32, (String, FeatureType)>);

impl FeatureMap {
    /// Read a `FeatureMap` from a file at given path.
    ///
    /// File should contain one feature definition per line, and be of the form:
    /// ```text
    /// <number>\t<name>\t<type>\n
    /// ```
    ///
    /// Type should be one of:
    /// * `i` - binary feature
    /// * `q` - quantitative feature
    /// * `int` - integer features
    ///
    /// E.g.:
    /// ```text
    /// 0   age int
    /// 1   is-parent?=yes  i
    /// 2   is-parent?=no   i
    /// 3   income  int
    /// ```
    pub fn from_file<P: AsRef<Path>>(path: P) -> io::Result<FeatureMap> {
        let file = File::open(path)?;
        let mut features: FeatureMap = FeatureMap(BTreeMap::new());

        for (i, line) in BufReader::new(&file).lines().enumerate() {
            let line = line?;
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() != 3 {
                let msg = format!(
                    "Unable to parse features from line {}, expected 3 tab separated values",
                    i + 1
                );
                return Err(io::Error::new(io::ErrorKind::InvalidData, msg));
            }

            assert_eq!(parts.len(), 3);
            let feature_num: u32 = match parts[0].parse() {
                Ok(num) => num,
                Err(err) => {
                    let msg = format!(
                        "Unable to parse features from line {}, could not parse feature number: {}",
                        i + 1,
                        err
                    );
                    return Err(io::Error::new(io::ErrorKind::InvalidData, msg));
                }
            };

            let feature_name = &parts[1];
            let feature_type = match FeatureType::from_str(&parts[2]) {
                Ok(feature_type) => feature_type,
                Err(msg) => {
                    let msg = format!("Unable to parse features from line {}: {}", i + 1, msg);
                    return Err(io::Error::new(io::ErrorKind::InvalidData, msg));
                }
            };
            features
                .0
                .insert(feature_num, (feature_name.to_string(), feature_type));
        }
        Ok(features)
    }
}

/// Indicates the type of a feature, used when dumping models as text.
pub enum FeatureType {
    /// Binary indicator feature.
    Binary,

    /// Quantitative feature (e.g. age, time, etc.), can be missing.
    Quantitative,

    /// Integer feature (when hinted, decision boundary will be integer).
    Integer,
}

impl FromStr for FeatureType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "i" => Ok(FeatureType::Binary),
            "q" => Ok(FeatureType::Quantitative),
            "int" => Ok(FeatureType::Integer),
            _ => Err(format!(
                "unrecognised feature type '{}', must be one of: 'i', 'q', 'int'",
                s
            )),
        }
    }
}

impl fmt::Display for FeatureType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let s = match self {
            FeatureType::Binary => "i",
            FeatureType::Quantitative => "q",
            FeatureType::Integer => "int",
        };
        write!(f, "{}", s)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::arr2;

    use super::*;
    use crate::parameters::{self, learning, tree};

    fn read_train_matrix() -> XGBResult<DMatrix> {
        DMatrix::load("data.csv?format=csv")
    }

    fn load_test_booster() -> Booster {
        let dmat = read_train_matrix().expect("Reading train matrix failed");
        Booster::new_with_cached_dmats(&BoosterParameters::default(), &[&dmat])
            .expect("Creating Booster failed")
    }

    #[test]
    fn set_booster_param() {
        let mut booster = load_test_booster();
        let res = booster.set_param("key", "value");
        assert!(res.is_ok());
    }

    #[test]
    fn load_rabit_version() {
        let version = load_test_booster().load_rabit_checkpoint().unwrap();
        assert_eq!(version, 0);
    }

    #[test]
    fn get_set_attr() {
        let mut booster = load_test_booster();
        let attr = booster
            .get_attribute("foo")
            .expect("Getting attribute failed");
        assert_eq!(attr, None);

        booster
            .set_attribute("foo", "bar")
            .expect("Setting attribute failed");
        let attr = booster
            .get_attribute("foo")
            .expect("Getting attribute failed");
        assert_eq!(attr, Some("bar".to_owned()));
    }

    #[test]
    fn save_and_load_from_buffer() {
        let dmat_train = DMatrix::load("agaricus.txt.train").unwrap();
        let mut booster =
            Booster::new_with_cached_dmats(&BoosterParameters::default(), &[&dmat_train]).unwrap();
        let attr = booster
            .get_attribute("foo")
            .expect("Getting attribute failed");
        assert_eq!(attr, None);

        booster
            .set_attribute("foo", "bar")
            .expect("Setting attribute failed");
        let attr = booster
            .get_attribute("foo")
            .expect("Getting attribute failed");
        assert_eq!(attr, Some("bar".to_owned()));

        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("test-xgboost-model");
        booster.save(&path).expect("saving booster");
        drop(booster);
        let bytes = std::fs::read(&path).expect("read saved booster file");
        let booster = Booster::load_buffer(&bytes[..]).expect("load booster from buffer");
        let attr = booster
            .get_attribute("foo")
            .expect("Getting attribute failed");
        assert_eq!(attr, Some("bar".to_owned()));
    }

    #[test]
    fn get_attribute_names() {
        let mut booster = load_test_booster();
        let attrs = booster
            .get_attribute_names()
            .expect("Getting attributes failed");
        assert_eq!(attrs, Vec::<String>::new());

        booster
            .set_attribute("foo", "bar")
            .expect("Setting attribute failed");
        booster
            .set_attribute("another", "another")
            .expect("Setting attribute failed");
        booster
            .set_attribute("4", "4")
            .expect("Setting attribute failed");
        booster
            .set_attribute("an even longer attribute name?", "")
            .expect("Setting attribute failed");

        let mut expected = vec!["foo", "another", "4", "an even longer attribute name?"];
        expected.sort();
        let mut attrs = booster
            .get_attribute_names()
            .expect("Getting attributes failed");
        attrs.sort();
        assert_eq!(attrs, expected);
    }

    #[test]
    fn predict() {
        let dmat_train = DMatrix::load("agaricus.txt.train?format=libsvm").unwrap();
        let dmat_test = DMatrix::load("agaricus.txt.test?format=libsvm").unwrap();

        let tree_params = tree::TreeBoosterParametersBuilder::default()
            .max_depth(2)
            .eta(1.0)
            .build()
            .unwrap();
        let learning_params = learning::LearningTaskParametersBuilder::default()
            .objective(learning::Objective::BinaryLogistic)
            .eval_metrics(learning::Metrics::Custom(vec![
                learning::EvaluationMetric::MAPCutNegative(4),
                learning::EvaluationMetric::LogLoss,
                learning::EvaluationMetric::BinaryErrorRate(0.5),
            ]))
            .build()
            .unwrap();
        let params = parameters::BoosterParametersBuilder::default()
            .booster_type(parameters::BoosterType::Tree(tree_params))
            .learning_params(learning_params)
            .verbose(false)
            .build()
            .unwrap();
        let mut booster =
            Booster::new_with_cached_dmats(&params, &[&dmat_train, &dmat_test]).unwrap();

        for i in 0..10 {
            booster.update(&dmat_train, i).expect("update failed");
        }

        println!("ping");

        let eps = 1e-6;

        let train_metrics = booster.evaluate(&dmat_train).unwrap();
        assert!(*train_metrics.get("logloss").unwrap() - 0.006634 < eps);
        assert!(*train_metrics.get("map@4-").unwrap() - 0.001274 < eps);

        let test_metrics = booster.evaluate(&dmat_test).unwrap();
        assert!(*test_metrics.get("logloss").unwrap() - 0.00692 < eps);
        assert!(*test_metrics.get("map@4-").unwrap() - 0.005155 < eps);

        let v = booster.predict(&dmat_test).unwrap();
        assert_eq!(v.len(), dmat_test.num_rows());

        // first 10 predictions
        let expected_start = [
            0.0050151693,
            0.9884467,
            0.0050151693,
            0.0050151693,
            0.026636455,
            0.11789363,
            0.9884467,
            0.01231471,
            0.9884467,
            0.00013656063,
        ];

        // last 10 predictions
        let expected_end = [
            0.002520344,
            0.00060917926,
            0.99881005,
            0.00060917926,
            0.00060917926,
            0.00060917926,
            0.00060917926,
            0.9981102,
            0.002855195,
            0.9981102,
        ];

        for (pred, expected) in v.iter().zip(&expected_start) {
            println!("predictions={}, expected={}", pred, expected);
            assert!(pred - expected < eps);
        }

        for (pred, expected) in v[v.len() - 10..].iter().zip(&expected_end) {
            println!("predictions={}, expected={}", pred, expected);
            assert!(pred - expected < eps);
        }
    }

    #[test]
    fn predict_leaf() {
        let dmat_train = DMatrix::load("agaricus.txt.train").unwrap();
        let dmat_test = DMatrix::load("agaricus.txt.test").unwrap();

        let tree_params = tree::TreeBoosterParametersBuilder::default()
            .max_depth(2)
            .eta(1.0)
            .build()
            .unwrap();
        let learning_params = learning::LearningTaskParametersBuilder::default()
            .objective(learning::Objective::BinaryLogistic)
            .eval_metrics(learning::Metrics::Custom(vec![
                learning::EvaluationMetric::LogLoss,
            ]))
            .build()
            .unwrap();
        let params = parameters::BoosterParametersBuilder::default()
            .booster_type(parameters::BoosterType::Tree(tree_params))
            .learning_params(learning_params)
            .verbose(false)
            .build()
            .unwrap();
        let mut booster =
            Booster::new_with_cached_dmats(&params, &[&dmat_train, &dmat_test]).unwrap();

        let num_rounds = 15;
        for i in 0..num_rounds {
            booster.update(&dmat_train, i).expect("update failed");
        }

        let (_preds, shape) = booster.predict_leaf(&dmat_test).unwrap();
        let num_samples = dmat_test.num_rows();
        assert_eq!(shape, (num_samples, num_rounds as usize));
    }

    #[test]
    fn predict_contributions() {
        let dmat_train = DMatrix::load("agaricus.txt.train").unwrap();
        let dmat_test = DMatrix::load("agaricus.txt.test").unwrap();

        let tree_params = tree::TreeBoosterParametersBuilder::default()
            .max_depth(2)
            .eta(1.0)
            .build()
            .unwrap();
        let learning_params = learning::LearningTaskParametersBuilder::default()
            .objective(learning::Objective::BinaryLogistic)
            .eval_metrics(learning::Metrics::Custom(vec![
                learning::EvaluationMetric::LogLoss,
            ]))
            .build()
            .unwrap();
        let params = parameters::BoosterParametersBuilder::default()
            .booster_type(parameters::BoosterType::Tree(tree_params))
            .learning_params(learning_params)
            .verbose(false)
            .build()
            .unwrap();
        let mut booster =
            Booster::new_with_cached_dmats(&params, &[&dmat_train, &dmat_test]).unwrap();

        let num_rounds = 5;
        for i in 0..num_rounds {
            booster.update(&dmat_train, i).expect("update failed");
        }

        let (_preds, shape) = booster.predict_contributions(&dmat_test).unwrap();
        let num_samples = dmat_test.num_rows();
        let num_features = dmat_train.num_cols();
        assert_eq!(shape, (num_samples, num_features + 1));
    }

    #[test]
    fn predict_interactions() {
        let dmat_train = DMatrix::load("agaricus.txt.train").unwrap();
        let dmat_test = DMatrix::load("agaricus.txt.test").unwrap();

        let tree_params = tree::TreeBoosterParametersBuilder::default()
            .max_depth(2)
            .eta(1.0)
            .build()
            .unwrap();
        let learning_params = learning::LearningTaskParametersBuilder::default()
            .objective(learning::Objective::BinaryLogistic)
            .eval_metrics(learning::Metrics::Custom(vec![
                learning::EvaluationMetric::LogLoss,
            ]))
            .build()
            .unwrap();
        let params = parameters::BoosterParametersBuilder::default()
            .booster_type(parameters::BoosterType::Tree(tree_params))
            .learning_params(learning_params)
            .verbose(false)
            .build()
            .unwrap();
        let mut booster =
            Booster::new_with_cached_dmats(&params, &[&dmat_train, &dmat_test]).unwrap();

        let num_rounds = 5;
        for i in 0..num_rounds {
            booster.update(&dmat_train, i).expect("update failed");
        }

        let (_preds, shape) = booster.predict_interactions(&dmat_test).unwrap();
        let num_samples = dmat_test.num_rows();
        let num_features = dmat_train.num_cols();
        assert_eq!(shape, (num_samples, num_features + 1, num_features + 1));
    }

    #[test]
    fn parse_eval_string() {
        let s = "[0]\ttrain-map@4-:0.5\ttrain-logloss:1.0\ttest-map@4-:0.25\ttest-logloss:0.75";
        let mut metrics = IndexMap::new();

        let mut train_metrics = IndexMap::new();
        train_metrics.insert("map@4-".to_owned(), 0.5);
        train_metrics.insert("logloss".to_owned(), 1.0);

        let mut test_metrics = IndexMap::new();
        test_metrics.insert("map@4-".to_owned(), 0.25);
        test_metrics.insert("logloss".to_owned(), 0.75);

        metrics.insert("train".to_owned(), train_metrics);
        metrics.insert("test".to_owned(), test_metrics);
        assert_eq!(Booster::parse_eval_string(s, &["train", "test"]), metrics);
    }

    #[test]
    fn pred_from_dmat() {
        let data_arr_2d = arr2(&[
            [
                8.32520000e+00,
                4.10000000e+01,
                6.98412698e+00,
                1.02380952e+00,
                3.22000000e+02,
                2.55555556e+00,
                3.78800000e+01,
                -1.22230000e+02,
            ],
            [
                8.30140000e+00,
                2.10000000e+01,
                6.23813708e+00,
                9.71880492e-01,
                2.40100000e+03,
                2.10984183e+00,
                3.78600000e+01,
                -1.22220000e+02,
            ],
            [
                7.25740000e+00,
                5.20000000e+01,
                8.28813559e+00,
                1.07344633e+00,
                4.96000000e+02,
                2.80225989e+00,
                3.78500000e+01,
                -1.22240000e+02,
            ],
            [
                5.64310000e+00,
                5.20000000e+01,
                5.81735160e+00,
                1.07305936e+00,
                5.58000000e+02,
                2.54794521e+00,
                3.78500000e+01,
                -1.22250000e+02,
            ],
            [
                3.84620000e+00,
                5.20000000e+01,
                6.28185328e+00,
                1.08108108e+00,
                5.65000000e+02,
                2.18146718e+00,
                3.78500000e+01,
                -1.22250000e+02,
            ],
            [
                4.03680000e+00,
                5.20000000e+01,
                4.76165803e+00,
                1.10362694e+00,
                4.13000000e+02,
                2.13989637e+00,
                3.78500000e+01,
                -1.22250000e+02,
            ],
            [
                3.65910000e+00,
                5.20000000e+01,
                4.93190661e+00,
                9.51361868e-01,
                1.09400000e+03,
                2.12840467e+00,
                3.78400000e+01,
                -1.22250000e+02,
            ],
            [
                3.12000000e+00,
                5.20000000e+01,
                4.79752705e+00,
                1.06182380e+00,
                1.15700000e+03,
                1.78825348e+00,
                3.78400000e+01,
                -1.22250000e+02,
            ],
            [
                2.08040000e+00,
                4.20000000e+01,
                4.29411765e+00,
                1.11764706e+00,
                1.20600000e+03,
                2.02689076e+00,
                3.78400000e+01,
                -1.22260000e+02,
            ],
            [
                3.69120000e+00,
                5.20000000e+01,
                4.97058824e+00,
                9.90196078e-01,
                1.55100000e+03,
                2.17226891e+00,
                3.78400000e+01,
                -1.22250000e+02,
            ],
            [
                3.20310000e+00,
                5.20000000e+01,
                5.47761194e+00,
                1.07960199e+00,
                9.10000000e+02,
                2.26368159e+00,
                3.78500000e+01,
                -1.22260000e+02,
            ],
            [
                3.27050000e+00,
                5.20000000e+01,
                4.77247956e+00,
                1.02452316e+00,
                1.50400000e+03,
                2.04904632e+00,
                3.78500000e+01,
                -1.22260000e+02,
            ],
            [
                3.07500000e+00,
                5.20000000e+01,
                5.32264957e+00,
                1.01282051e+00,
                1.09800000e+03,
                2.34615385e+00,
                3.78500000e+01,
                -1.22260000e+02,
            ],
            [
                2.67360000e+00,
                5.20000000e+01,
                4.00000000e+00,
                1.09770115e+00,
                3.45000000e+02,
                1.98275862e+00,
                3.78400000e+01,
                -1.22260000e+02,
            ],
            [
                1.91670000e+00,
                5.20000000e+01,
                4.26290323e+00,
                1.00967742e+00,
                1.21200000e+03,
                1.95483871e+00,
                3.78500000e+01,
                -1.22260000e+02,
            ],
            [
                2.12500000e+00,
                5.00000000e+01,
                4.24242424e+00,
                1.07196970e+00,
                6.97000000e+02,
                2.64015152e+00,
                3.78500000e+01,
                -1.22260000e+02,
            ],
            [
                2.77500000e+00,
                5.20000000e+01,
                5.93957704e+00,
                1.04833837e+00,
                7.93000000e+02,
                2.39577039e+00,
                3.78500000e+01,
                -1.22270000e+02,
            ],
            [
                2.12020000e+00,
                5.20000000e+01,
                4.05280528e+00,
                9.66996700e-01,
                6.48000000e+02,
                2.13861386e+00,
                3.78500000e+01,
                -1.22270000e+02,
            ],
            [
                1.99110000e+00,
                5.00000000e+01,
                5.34367542e+00,
                1.08591885e+00,
                9.90000000e+02,
                2.36276850e+00,
                3.78400000e+01,
                -1.22260000e+02,
            ],
            [
                2.60330000e+00,
                5.20000000e+01,
                5.46545455e+00,
                1.08363636e+00,
                6.90000000e+02,
                2.50909091e+00,
                3.78400000e+01,
                -1.22270000e+02,
            ],
            [
                1.35780000e+00,
                4.00000000e+01,
                4.52409639e+00,
                1.10843373e+00,
                4.09000000e+02,
                2.46385542e+00,
                3.78500000e+01,
                -1.22270000e+02,
            ],
            [
                1.71350000e+00,
                4.20000000e+01,
                4.47814208e+00,
                1.00273224e+00,
                9.29000000e+02,
                2.53825137e+00,
                3.78500000e+01,
                -1.22270000e+02,
            ],
            [
                1.72500000e+00,
                5.20000000e+01,
                5.09623431e+00,
                1.13179916e+00,
                1.01500000e+03,
                2.12343096e+00,
                3.78400000e+01,
                -1.22270000e+02,
            ],
            [
                2.18060000e+00,
                5.20000000e+01,
                5.19384615e+00,
                1.03692308e+00,
                8.53000000e+02,
                2.62461538e+00,
                3.78400000e+01,
                -1.22270000e+02,
            ],
            [
                2.60000000e+00,
                5.20000000e+01,
                5.27014218e+00,
                1.03554502e+00,
                1.00600000e+03,
                2.38388626e+00,
                3.78400000e+01,
                -1.22270000e+02,
            ],
            [
                2.40380000e+00,
                4.10000000e+01,
                4.49579832e+00,
                1.03361345e+00,
                3.17000000e+02,
                2.66386555e+00,
                3.78500000e+01,
                -1.22280000e+02,
            ],
            [
                2.45970000e+00,
                4.90000000e+01,
                4.72803347e+00,
                1.02092050e+00,
                6.07000000e+02,
                2.53974895e+00,
                3.78500000e+01,
                -1.22280000e+02,
            ],
            [
                1.80800000e+00,
                5.20000000e+01,
                4.78085642e+00,
                1.06045340e+00,
                1.10200000e+03,
                2.77581864e+00,
                3.78500000e+01,
                -1.22280000e+02,
            ],
            [
                1.64240000e+00,
                5.00000000e+01,
                4.40169133e+00,
                1.04016913e+00,
                1.13100000e+03,
                2.39112051e+00,
                3.78400000e+01,
                -1.22280000e+02,
            ],
            [
                1.68750000e+00,
                5.20000000e+01,
                4.70322581e+00,
                1.03225806e+00,
                3.95000000e+02,
                2.54838710e+00,
                3.78400000e+01,
                -1.22280000e+02,
            ],
            [
                1.92740000e+00,
                4.90000000e+01,
                5.06878307e+00,
                1.18253968e+00,
                8.63000000e+02,
                2.28306878e+00,
                3.78400000e+01,
                -1.22280000e+02,
            ],
            [
                1.96150000e+00,
                5.20000000e+01,
                4.88208617e+00,
                1.09070295e+00,
                1.16800000e+03,
                2.64852608e+00,
                3.78400000e+01,
                -1.22280000e+02,
            ],
            [
                1.79690000e+00,
                4.80000000e+01,
                5.73731343e+00,
                1.22089552e+00,
                1.02600000e+03,
                3.06268657e+00,
                3.78400000e+01,
                -1.22270000e+02,
            ],
            [
                1.37500000e+00,
                4.90000000e+01,
                5.03039514e+00,
                1.11246201e+00,
                7.54000000e+02,
                2.29179331e+00,
                3.78300000e+01,
                -1.22270000e+02,
            ],
            [
                2.73030000e+00,
                5.10000000e+01,
                4.97201493e+00,
                1.07089552e+00,
                1.25800000e+03,
                2.34701493e+00,
                3.78300000e+01,
                -1.22270000e+02,
            ],
            [
                1.48610000e+00,
                4.90000000e+01,
                4.60227273e+00,
                1.06818182e+00,
                5.70000000e+02,
                2.15909091e+00,
                3.78300000e+01,
                -1.22270000e+02,
            ],
            [
                1.09720000e+00,
                4.80000000e+01,
                4.80748663e+00,
                1.15508021e+00,
                9.87000000e+02,
                2.63903743e+00,
                3.78300000e+01,
                -1.22270000e+02,
            ],
            [
                1.41030000e+00,
                5.20000000e+01,
                3.74937965e+00,
                9.67741935e-01,
                9.01000000e+02,
                2.23573201e+00,
                3.78300000e+01,
                -1.22280000e+02,
            ],
            [
                3.48000000e+00,
                5.20000000e+01,
                4.75728155e+00,
                1.06796117e+00,
                6.89000000e+02,
                2.22977346e+00,
                3.78300000e+01,
                -1.22260000e+02,
            ],
            [
                2.58980000e+00,
                5.20000000e+01,
                3.49425287e+00,
                1.02729885e+00,
                1.37700000e+03,
                1.97844828e+00,
                3.78300000e+01,
                -1.22260000e+02,
            ],
            [
                2.09780000e+00,
                5.20000000e+01,
                4.21518987e+00,
                1.06075949e+00,
                9.46000000e+02,
                2.39493671e+00,
                3.78300000e+01,
                -1.22260000e+02,
            ],
            [
                1.28520000e+00,
                5.10000000e+01,
                3.75903614e+00,
                1.24899598e+00,
                5.17000000e+02,
                2.07630522e+00,
                3.78300000e+01,
                -1.22260000e+02,
            ],
            [
                1.02500000e+00,
                4.90000000e+01,
                3.77248677e+00,
                1.06878307e+00,
                4.62000000e+02,
                2.44444444e+00,
                3.78400000e+01,
                -1.22260000e+02,
            ],
            [
                3.96430000e+00,
                5.20000000e+01,
                4.79797980e+00,
                1.02020202e+00,
                4.67000000e+02,
                2.35858586e+00,
                3.78400000e+01,
                -1.22260000e+02,
            ],
            [
                3.01250000e+00,
                5.20000000e+01,
                4.94178082e+00,
                1.06506849e+00,
                6.60000000e+02,
                2.26027397e+00,
                3.78300000e+01,
                -1.22260000e+02,
            ],
            [
                2.67680000e+00,
                5.20000000e+01,
                4.33507853e+00,
                1.09947644e+00,
                7.18000000e+02,
                1.87958115e+00,
                3.78300000e+01,
                -1.22260000e+02,
            ],
            [
                2.02600000e+00,
                5.00000000e+01,
                3.70065789e+00,
                1.05921053e+00,
                6.16000000e+02,
                2.02631579e+00,
                3.78300000e+01,
                -1.22260000e+02,
            ],
            [
                1.73480000e+00,
                4.30000000e+01,
                3.98023715e+00,
                1.23320158e+00,
                5.58000000e+02,
                2.20553360e+00,
                3.78200000e+01,
                -1.22270000e+02,
            ],
            [
                9.50600000e-01,
                4.00000000e+01,
                3.90000000e+00,
                1.21875000e+00,
                4.23000000e+02,
                2.64375000e+00,
                3.78200000e+01,
                -1.22260000e+02,
            ],
            [
                1.77500000e+00,
                4.00000000e+01,
                2.68750000e+00,
                1.06534091e+00,
                7.00000000e+02,
                1.98863636e+00,
                3.78200000e+01,
                -1.22270000e+02,
            ],
            [
                9.21800000e-01,
                2.10000000e+01,
                2.04566210e+00,
                1.03424658e+00,
                7.35000000e+02,
                1.67808219e+00,
                3.78200000e+01,
                -1.22270000e+02,
            ],
            [
                1.50450000e+00,
                4.30000000e+01,
                4.58968059e+00,
                1.12039312e+00,
                1.06100000e+03,
                2.60687961e+00,
                3.78200000e+01,
                -1.22270000e+02,
            ],
            [
                1.11080000e+00,
                4.10000000e+01,
                4.47361111e+00,
                1.18472222e+00,
                1.95900000e+03,
                2.72083333e+00,
                3.78200000e+01,
                -1.22270000e+02,
            ],
            [
                1.24750000e+00,
                5.20000000e+01,
                4.07500000e+00,
                1.14000000e+00,
                1.16200000e+03,
                2.90500000e+00,
                3.78200000e+01,
                -1.22270000e+02,
            ],
            [
                1.60980000e+00,
                5.20000000e+01,
                5.02145923e+00,
                1.00858369e+00,
                7.01000000e+02,
                3.00858369e+00,
                3.78200000e+01,
                -1.22280000e+02,
            ],
            [
                1.41130000e+00,
                5.20000000e+01,
                4.29545455e+00,
                1.10454545e+00,
                5.76000000e+02,
                2.61818182e+00,
                3.78200000e+01,
                -1.22280000e+02,
            ],
            [
                1.50570000e+00,
                5.20000000e+01,
                4.77992278e+00,
                1.11196911e+00,
                6.22000000e+02,
                2.40154440e+00,
                3.78200000e+01,
                -1.22280000e+02,
            ],
            [
                8.17200000e-01,
                5.20000000e+01,
                6.10245902e+00,
                1.37295082e+00,
                7.28000000e+02,
                2.98360656e+00,
                3.78200000e+01,
                -1.22280000e+02,
            ],
            [
                1.21710000e+00,
                5.20000000e+01,
                4.56250000e+00,
                1.12171053e+00,
                1.07400000e+03,
                3.53289474e+00,
                3.78200000e+01,
                -1.22280000e+02,
            ],
            [
                2.56250000e+00,
                2.00000000e+00,
                2.77192982e+00,
                7.54385965e-01,
                9.40000000e+01,
                1.64912281e+00,
                3.78200000e+01,
                -1.22290000e+02,
            ],
            [
                3.39290000e+00,
                5.20000000e+01,
                5.99465241e+00,
                1.12834225e+00,
                5.54000000e+02,
                2.96256684e+00,
                3.78300000e+01,
                -1.22290000e+02,
            ],
            [
                6.11830000e+00,
                4.90000000e+01,
                5.86956522e+00,
                1.26086957e+00,
                8.60000000e+01,
                3.73913043e+00,
                3.78200000e+01,
                -1.22290000e+02,
            ],
            [
                9.01100000e-01,
                5.00000000e+01,
                6.22950820e+00,
                1.55737705e+00,
                3.77000000e+02,
                3.09016393e+00,
                3.78100000e+01,
                -1.22290000e+02,
            ],
            [
                1.19100000e+00,
                5.20000000e+01,
                7.69811321e+00,
                1.49056604e+00,
                5.21000000e+02,
                3.27672956e+00,
                3.78100000e+01,
                -1.22300000e+02,
            ],
            [
                2.59380000e+00,
                4.80000000e+01,
                6.22556391e+00,
                1.36842105e+00,
                3.92000000e+02,
                2.94736842e+00,
                3.78100000e+01,
                -1.22300000e+02,
            ],
            [
                1.16670000e+00,
                5.20000000e+01,
                5.40106952e+00,
                1.11764706e+00,
                6.04000000e+02,
                3.22994652e+00,
                3.78100000e+01,
                -1.22300000e+02,
            ],
            [
                8.05600000e-01,
                4.80000000e+01,
                4.38253012e+00,
                1.06626506e+00,
                7.88000000e+02,
                2.37349398e+00,
                3.78100000e+01,
                -1.22300000e+02,
            ],
            [
                2.60940000e+00,
                5.20000000e+01,
                6.98639456e+00,
                1.65986395e+00,
                4.92000000e+02,
                3.34693878e+00,
                3.78000000e+01,
                -1.22290000e+02,
            ],
            [
                1.85160000e+00,
                5.20000000e+01,
                6.97560976e+00,
                1.32926829e+00,
                2.74000000e+02,
                3.34146341e+00,
                3.78100000e+01,
                -1.22300000e+02,
            ],
            [
                9.80200000e-01,
                4.60000000e+01,
                4.58428805e+00,
                1.05400982e+00,
                1.82300000e+03,
                2.98363339e+00,
                3.78100000e+01,
                -1.22290000e+02,
            ],
            [
                1.77190000e+00,
                2.60000000e+01,
                6.04724409e+00,
                1.19685039e+00,
                3.92000000e+02,
                3.08661417e+00,
                3.78100000e+01,
                -1.22290000e+02,
            ],
            [
                7.28600000e-01,
                4.60000000e+01,
                3.37545126e+00,
                1.07220217e+00,
                5.82000000e+02,
                2.10108303e+00,
                3.78100000e+01,
                -1.22290000e+02,
            ],
            [
                1.75000000e+00,
                4.90000000e+01,
                5.55263158e+00,
                1.34210526e+00,
                5.60000000e+02,
                3.68421053e+00,
                3.78100000e+01,
                -1.22290000e+02,
            ],
            [
                4.99900000e-01,
                4.60000000e+01,
                1.71428571e+00,
                5.71428571e-01,
                1.80000000e+01,
                2.57142857e+00,
                3.78100000e+01,
                -1.22290000e+02,
            ],
            [
                2.48300000e+00,
                2.00000000e+01,
                6.27819549e+00,
                1.21052632e+00,
                2.90000000e+02,
                2.18045113e+00,
                3.78100000e+01,
                -1.22290000e+02,
            ],
            [
                9.24100000e-01,
                1.70000000e+01,
                2.81776765e+00,
                1.05239180e+00,
                7.62000000e+02,
                1.73576310e+00,
                3.78100000e+01,
                -1.22280000e+02,
            ],
            [
                2.44640000e+00,
                3.60000000e+01,
                5.72495088e+00,
                1.10412574e+00,
                1.23600000e+03,
                2.42829077e+00,
                3.78100000e+01,
                -1.22280000e+02,
            ],
            [
                1.11110000e+00,
                1.90000000e+01,
                5.83091787e+00,
                1.17391304e+00,
                7.21000000e+02,
                3.48309179e+00,
                3.78100000e+01,
                -1.22280000e+02,
            ],
            [
                8.02600000e-01,
                2.30000000e+01,
                5.36923077e+00,
                1.15076923e+00,
                1.05400000e+03,
                3.24307692e+00,
                3.78100000e+01,
                -1.22290000e+02,
            ],
            [
                2.01140000e+00,
                3.80000000e+01,
                4.41290323e+00,
                1.13548387e+00,
                3.44000000e+02,
                2.21935484e+00,
                3.78000000e+01,
                -1.22280000e+02,
            ],
            [
                1.50000000e+00,
                1.70000000e+01,
                3.19723183e+00,
                1.00000000e+00,
                6.09000000e+02,
                2.10726644e+00,
                3.78100000e+01,
                -1.22280000e+02,
            ],
            [
                1.16670000e+00,
                5.20000000e+01,
                3.75000000e+00,
                1.00000000e+00,
                1.83000000e+02,
                3.26785714e+00,
                3.78100000e+01,
                -1.22270000e+02,
            ],
            [
                1.52080000e+00,
                5.20000000e+01,
                3.90804598e+00,
                1.11494253e+00,
                2.00000000e+02,
                2.29885057e+00,
                3.78100000e+01,
                -1.22280000e+02,
            ],
            [
                8.07500000e-01,
                5.20000000e+01,
                2.49032258e+00,
                1.05806452e+00,
                3.46000000e+02,
                2.23225806e+00,
                3.78100000e+01,
                -1.22280000e+02,
            ],
            [
                1.80880000e+00,
                3.50000000e+01,
                5.60946746e+00,
                1.08875740e+00,
                4.67000000e+02,
                2.76331361e+00,
                3.78100000e+01,
                -1.22280000e+02,
            ],
            [
                2.40830000e+00,
                5.20000000e+01,
                6.72173913e+00,
                1.24347826e+00,
                3.77000000e+02,
                3.27826087e+00,
                3.78100000e+01,
                -1.22280000e+02,
            ],
            [
                9.77000000e-01,
                4.00000000e+01,
                2.31578947e+00,
                1.18684211e+00,
                5.82000000e+02,
                1.53157895e+00,
                3.78100000e+01,
                -1.22270000e+02,
            ],
            [
                7.60000000e-01,
                1.00000000e+01,
                2.65151515e+00,
                1.05454545e+00,
                5.46000000e+02,
                1.65454545e+00,
                3.78100000e+01,
                -1.22270000e+02,
            ],
            [
                9.72200000e-01,
                1.00000000e+01,
                2.69230769e+00,
                1.07692308e+00,
                1.25000000e+02,
                3.20512821e+00,
                3.78000000e+01,
                -1.22270000e+02,
            ],
            [
                1.24340000e+00,
                5.20000000e+01,
                2.92941176e+00,
                9.17647059e-01,
                3.96000000e+02,
                4.65882353e+00,
                3.78000000e+01,
                -1.22270000e+02,
            ],
            [
                2.09380000e+00,
                1.60000000e+01,
                2.74585635e+00,
                1.08287293e+00,
                8.00000000e+02,
                2.20994475e+00,
                3.78000000e+01,
                -1.22270000e+02,
            ],
            [
                8.66800000e-01,
                5.20000000e+01,
                2.44318182e+00,
                9.88636364e-01,
                9.04000000e+02,
                1.02727273e+01,
                3.78000000e+01,
                -1.22280000e+02,
            ],
            [
                7.50000000e-01,
                5.20000000e+01,
                2.82352941e+00,
                9.11764706e-01,
                1.91000000e+02,
                5.61764706e+00,
                3.78000000e+01,
                -1.22280000e+02,
            ],
            [
                2.63540000e+00,
                2.70000000e+01,
                3.49337748e+00,
                1.14900662e+00,
                7.18000000e+02,
                2.37748344e+00,
                3.77900000e+01,
                -1.22270000e+02,
            ],
            [
                1.84770000e+00,
                3.90000000e+01,
                3.67237687e+00,
                1.33404711e+00,
                1.32700000e+03,
                2.84154176e+00,
                3.78000000e+01,
                -1.22270000e+02,
            ],
            [
                2.00960000e+00,
                3.60000000e+01,
                2.29401636e+00,
                1.06629359e+00,
                3.46900000e+03,
                1.49332759e+00,
                3.78000000e+01,
                -1.22260000e+02,
            ],
            [
                2.83450000e+00,
                3.10000000e+01,
                3.89491525e+00,
                1.12796610e+00,
                2.04800000e+03,
                1.73559322e+00,
                3.78200000e+01,
                -1.22260000e+02,
            ],
            [
                2.00620000e+00,
                2.90000000e+01,
                3.68131868e+00,
                1.17582418e+00,
                2.02000000e+02,
                2.21978022e+00,
                3.78100000e+01,
                -1.22260000e+02,
            ],
            [
                1.21850000e+00,
                2.20000000e+01,
                2.94560000e+00,
                1.01600000e+00,
                2.02400000e+03,
                1.61920000e+00,
                3.78200000e+01,
                -1.22260000e+02,
            ],
            [
                2.61040000e+00,
                3.70000000e+01,
                3.70714286e+00,
                1.10714286e+00,
                1.83800000e+03,
                1.87551020e+00,
                3.78200000e+01,
                -1.22260000e+02,
            ],
        ]);

        let target_vec = [
            4.526, 3.585, 3.521, 3.413, 3.422, 2.697, 2.992, 2.414, 2.267, 2.611, 2.815, 2.418,
            2.135, 1.913, 1.592, 1.4, 1.525, 1.555, 1.587, 1.629, 1.475, 1.598, 1.139, 0.997,
            1.326, 1.075, 0.938, 1.055, 1.089, 1.32, 1.223, 1.152, 1.104, 1.049, 1.097, 0.972,
            1.045, 1.039, 1.914, 1.76, 1.554, 1.5, 1.188, 1.888, 1.844, 1.823, 1.425, 1.375, 1.875,
            1.125, 1.719, 0.938, 0.975, 1.042, 0.875, 0.831, 0.875, 0.853, 0.803, 0.6, 0.757, 0.75,
            0.861, 0.761, 0.735, 0.784, 0.844, 0.813, 0.85, 1.292, 0.825, 0.952, 0.75, 0.675,
            1.375, 1.775, 1.021, 1.083, 1.125, 1.313, 1.625, 1.125, 1.125, 1.375, 1.188, 0.982,
            1.188, 1.625, 1.375, 5.00001, 1.625, 1.375, 1.625, 1.875, 1.792, 1.3, 1.838, 1.25, 1.7,
            1.931,
        ];

        // define information needed for xgboost
        let strides_ax_0 = data_arr_2d.strides()[0] as usize;
        let strides_ax_1 = data_arr_2d.strides()[1] as usize;
        let byte_size_ax_0 = std::mem::size_of::<f64>() * strides_ax_0;
        let byte_size_ax_1 = std::mem::size_of::<f64>() * strides_ax_1;

        // get xgboost style matrices
        let mut xg_matrix = DMatrix::from_col_major_f64(
            data_arr_2d.as_slice_memory_order().unwrap(),
            byte_size_ax_0,
            byte_size_ax_1,
            100,
            9,
        )
        .unwrap();

        // set labels
        // TODO: make more generic

        let lbls: Vec<f32> = target_vec.iter().map(|elem| *elem as f32).collect();
        xg_matrix.set_labels(lbls.as_slice()).unwrap();

        // ------------------------------------------------------
        // start training

        let mut initial_training_config: HashMap<&str, &str> = HashMap::new();

        initial_training_config.insert("validate_parameters", "1");
        initial_training_config.insert("process_type", "default");
        initial_training_config.insert("tree_method", "hist");
        initial_training_config.insert("eval_metric", "rmse");
        initial_training_config.insert("max_depth", "3");

        let evals = &[(&xg_matrix, "train")];
        let bst = Booster::train(
            Some(evals),
            &xg_matrix,
            initial_training_config,
            None, // <- No old model yet
        )
        .unwrap();

        let test_data_arr_2d = arr2(&[
            [
                1.91000000e+00,
                4.60000000e+01,
                5.00000000e+00,
                1.00413223e+00,
                5.23000000e+02,
                2.16115702e+00,
                3.93600000e+01,
                -1.21700000e+02,
                6.39000000e-01,
            ],
            [
                2.04740000e+00,
                3.70000000e+01,
                4.95744681e+00,
                1.05319149e+00,
                1.50500000e+03,
                3.20212766e+00,
                3.93600000e+01,
                -1.21700000e+02,
                5.60000000e-01,
            ],
            [
                1.83550000e+00,
                3.40000000e+01,
                5.10303030e+00,
                1.12727273e+00,
                6.35000000e+02,
                3.84848485e+00,
                3.93600000e+01,
                -1.21690000e+02,
                6.30000000e-01,
            ],
            [
                2.32430000e+00,
                2.70000000e+01,
                6.34718826e+00,
                1.06356968e+00,
                1.10000000e+03,
                2.68948655e+00,
                3.93800000e+01,
                -1.21740000e+02,
                8.55000000e-01,
            ],
            [
                2.52590000e+00,
                3.00000000e+01,
                5.50810811e+00,
                1.03783784e+00,
                5.01000000e+02,
                2.70810811e+00,
                3.93300000e+01,
                -1.21800000e+02,
                8.13000000e-01,
            ],
            [
                2.28130000e+00,
                2.10000000e+01,
                5.20727273e+00,
                1.03272727e+00,
                8.62000000e+02,
                3.13454545e+00,
                3.94200000e+01,
                -1.21710000e+02,
                5.76000000e-01,
            ],
            [
                2.17280000e+00,
                2.20000000e+01,
                5.61609907e+00,
                1.05882353e+00,
                9.41000000e+02,
                2.91331269e+00,
                3.94100000e+01,
                -1.21710000e+02,
                5.94000000e-01,
            ],
            [
                2.49430000e+00,
                2.90000000e+01,
                5.05089820e+00,
                9.79041916e-01,
                8.64000000e+02,
                2.58682635e+00,
                3.94000000e+01,
                -1.21750000e+02,
                8.19000000e-01,
            ],
            [
                3.39290000e+00,
                3.90000000e+01,
                6.65662651e+00,
                1.08433735e+00,
                4.08000000e+02,
                2.45783133e+00,
                3.94800000e+01,
                -1.21790000e+02,
                8.21000000e-01,
            ],
            [
                2.38160000e+00,
                1.60000000e+01,
                6.05595409e+00,
                1.12051650e+00,
                1.51600000e+03,
                2.17503587e+00,
                3.81500000e+01,
                -1.20460000e+02,
                1.16000000e+00,
            ],
            [
                2.50000000e+00,
                1.00000000e+01,
                5.38144330e+00,
                1.11683849e+00,
                7.85000000e+02,
                2.69759450e+00,
                3.81200000e+01,
                -1.20550000e+02,
                1.16100000e+00,
            ],
            [
                2.36540000e+00,
                3.40000000e+01,
                5.59063136e+00,
                1.13849287e+00,
                1.15000000e+03,
                2.34215886e+00,
                3.80900000e+01,
                -1.20560000e+02,
                9.49000000e-01,
            ],
            [
                2.90630000e+00,
                2.70000000e+01,
                6.02512563e+00,
                1.12562814e+00,
                4.63000000e+02,
                2.32663317e+00,
                3.80700000e+01,
                -1.20550000e+02,
                9.22000000e-01,
            ],
            [
                2.28750000e+00,
                3.70000000e+01,
                5.25714286e+00,
                1.05714286e+00,
                3.39000000e+02,
                2.42142857e+00,
                3.80700000e+01,
                -1.20540000e+02,
                7.99000000e-01,
            ],
            [
                2.65280000e+00,
                9.00000000e+00,
                8.01075269e+00,
                1.58602151e+00,
                2.23300000e+03,
                2.40107527e+00,
                3.79700000e+01,
                -1.20670000e+02,
                1.33000000e+00,
            ],
            [
                3.00000000e+00,
                1.60000000e+01,
                6.11056911e+00,
                1.16260163e+00,
                1.77700000e+03,
                2.88943089e+00,
                3.80900000e+01,
                -1.20460000e+02,
                1.22600000e+00,
            ],
            [
                2.98210000e+00,
                1.90000000e+01,
                5.27894737e+00,
                1.23684211e+00,
                5.38000000e+02,
                2.83157895e+00,
                3.82400000e+01,
                -1.20790000e+02,
                9.04000000e-01,
            ],
            [
                2.04720000e+00,
                1.60000000e+01,
                5.93155894e+00,
                1.21863118e+00,
                1.31900000e+03,
                2.50760456e+00,
                3.82000000e+01,
                -1.20900000e+02,
                9.32000000e-01,
            ],
            [
                4.01090000e+00,
                8.00000000e+00,
                5.57417582e+00,
                1.06318681e+00,
                1.00000000e+03,
                2.74725275e+00,
                3.81600000e+01,
                -1.20880000e+02,
                1.25900000e+00,
            ],
            [
                3.63600000e+00,
                9.00000000e+00,
                5.99498328e+00,
                1.13712375e+00,
                1.80000000e+03,
                3.01003344e+00,
                3.81100000e+01,
                -1.20910000e+02,
                1.33100000e+00,
            ],
        ]);

        let strides_ax_0 = test_data_arr_2d.strides()[0] as usize;
        let strides_ax_1 = test_data_arr_2d.strides()[1] as usize;
        let byte_size_ax_0 = std::mem::size_of::<f64>() * strides_ax_0;
        let byte_size_ax_1 = std::mem::size_of::<f64>() * strides_ax_1;

        // get xgboost style matrices
        let test_data = DMatrix::from_col_major_f64(
            test_data_arr_2d.as_slice_memory_order().unwrap(),
            byte_size_ax_0,
            byte_size_ax_1,
            20,
            9,
        )
        .unwrap();

        let mut out_dim: u64 = 10;
        let result = bst
            .predict_from_dmat(&test_data, &[20, 9], &mut out_dim)
            .unwrap();
        println!("result: {:?}", result);
    }
}
