use crate::dmatrix::DMatrix;
use crate::error::XGBError;
use indexmap::IndexMap;
use libc;
use std::collections::{BTreeMap, HashMap};
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
        keys: Vec<&str>,
        values: Vec<&str>,
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
        booster.set_param_from_json(keys, values);
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

    pub fn my_train(
        evaluation_sets: Option<&[(&DMatrix, &str)]>,
        dtrain: &DMatrix,
        keys: Vec<&str>,
        values: Vec<&str>,
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
            if let Some(mut booster) = bst {
                booster.set_param_from_json(keys, values);
                booster
            } else {
                let bst = Booster::new_with_json_config(&cached_dmats, keys, values)?;
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

        bst.save_config();
        Ok(bst)
    }

    /// Convenience function for creating/training a new Booster.
    ///
    /// This does the following:
    ///
    /// 1. create a new Booster model with given parameters
    /// 2. train the model with given DMatrix
    /// 3. print out evaluation results for each training round
    /// 4. return trained Booster
    ///
    /// * `params` - training parameters
    /// * `dtrain` - matrix to train Booster with
    /// * `num_boost_round` - number of training iterations
    /// * `eval_sets` - list of datasets to evaluate after each boosting round
    pub fn train(params: &TrainingParameters) -> XGBResult<Self> {
        let cached_dmats = {
            let mut dmats = vec![params.dtrain];
            if let Some(eval_sets) = params.evaluation_sets {
                for (dmat, _) in eval_sets {
                    dmats.push(*dmat);
                }
            }
            dmats
        };

        let mut bst = Booster::new_with_cached_dmats(&params.booster_params, &cached_dmats)?;
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

    fn set_param_from_json(&mut self, keys: Vec<&str>, values: Vec<&str>) {
        for (k, v) in zip(keys, values) {
            let name = ffi::CString::new(k).unwrap();
            let value = ffi::CString::new(v).unwrap();

            println!("setting {:?} to {:?}", k, v);

            let setting_ok = unsafe {
                xgboost_bib::XGBoosterSetParam(self.handle, name.as_ptr(), value.as_ptr())
            };
            println!("setting ok? {:?}", setting_ok);
        }
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
    fn inc_train() {}
}
