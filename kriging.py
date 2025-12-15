import numpy as np
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging
from skgstat import Variogram
import traceback
from typing import Dict, Any, List
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


class Interpolator:
    """插值计算工具（自动选择方法 → 检验正态性 → 选择最优变异函数 → 插值并返回网格）"""
    def __init__(self):
        self.basin_union = None

    def set_basin_boundary(self, basin_geom):
        self.basin_union = basin_geom

    # ================== 数据特征分析 ==================
    def suggest_kriging_method(self, lons, lats, values, alpha=0.05):
        """
        根据专业规则自动建议插值方法。
        """
        mask = np.isfinite(values)
        lons = np.array(lons)[mask]
        lats = np.array(lats)[mask]
        values = np.array(values)[mask]
        
        n_points = len(values)
        metrics = {"n_points": n_points}

        # 规则 1: 样本量过少
        if n_points < 20:
            return {"suggestion": "idw", "reason": f"数据量严重不足 ({n_points} < 20)，任何复杂模型都不可靠，IDW 是唯一稳健的选择", "metrics": metrics}

        # --- 数据量 >= 20，开始进行统计检验 ---

        # 1. 检验空间趋势
        try:
            corr_x, p_x = stats.pearsonr(values, lons)
        except Exception:
            corr_x, p_x = 0.0, 1.0
        try:
            corr_y, p_y = stats.pearsonr(values, lats)
        except Exception:
            corr_y, p_y = 0.0, 1.0
        trend_strength = max(abs(corr_x), abs(corr_y))
        is_trend_significant = p_x < alpha or p_y < alpha
        metrics.update({"corr_x": corr_x, "p_x": p_x, "corr_y": corr_y, "p_y": p_y, "trend_strength": trend_strength})

        # 规则 2: IDW - 空间关联性弱
        if trend_strength < 0.2:
            reason = "数据点分布相对均匀，空间关联性弱"
            return {"suggestion": "idw", "reason": reason, "metrics": metrics}
            
        # 规则 3: 趋势面插值 - 强全局趋势
        if trend_strength > 0.6 and is_trend_significant:
            reason = "数据存在可通过多项式拟合的、明显的全局性空间趋势"
            return {"suggestion": "trend", "reason": reason, "metrics": metrics}

        # 2. 检验平稳性 (正态性 & 变异性)
        try:
            _, shapiro_p = stats.shapiro(values)
        except Exception:
            shapiro_p = 1.0
        skewness = float(stats.skew(values))
        cv = float(np.std(values) / (np.mean(values) + 1e-9))
        metrics.update({"shapiro_p": shapiro_p, "skewness": skewness, "cv": cv})
        
        is_stationary = (trend_strength <= 0.3) and (shapiro_p >= alpha) and (cv <= 1.0)

        # 规则 4: 泛克里金 - 违反平稳性
        if not is_stationary:
            reason = "数据存在显著空间趋势（漂移），违反平稳性假设"
            return {"suggestion": "uk", "reason": reason, "metrics": metrics}

        # 规则 5: 普通克里金 - 满足平稳性
        if is_stationary:
            reason = "数据量充足，满足平稳性假设，且存在可量化的空间自相关"
            return {"suggestion": "ok", "reason": reason, "metrics": metrics}

        # 默认回退
        return {"suggestion": "ok", "reason": "默认选择：数据基本满足平稳性假设", "metrics": metrics}

    # ================== 工具函数 ==================
    def _trend_surface_interpolate(self, lons, lats, values, grid_x, grid_y, degree=2):
        """趋势面插值"""
        coords = np.c_[lons, lats]
        
        # 创建多项式特征
        poly = PolynomialFeatures(degree=degree)
        coords_poly = poly.fit_transform(coords)
        
        # 拟合线性模型
        model = LinearRegression()
        model.fit(coords_poly, values)
        
        # 在网格上预测
        grid_coords = np.c_[grid_x.ravel(), grid_y.ravel()]
        grid_coords_poly = poly.transform(grid_coords)
        z_pred = model.predict(grid_coords_poly)
        
        return z_pred.reshape(grid_x.shape)

    def _idw_interpolate(self, lons, lats, values, grid_x, grid_y, power=2, epsilon=1e-6):
        """反距离加权插值（IDW）"""
        ny, nx = grid_x.shape
        z = np.full((ny, nx), np.nan)
        
        # 展平网格以便迭代
        grid_points = np.c_[grid_x.ravel(), grid_y.ravel()]
        
        for i, (gx, gy) in enumerate(grid_points):
            distances = np.sqrt((lons - gx)**2 + (lats - gy)**2)
            
            # 处理距离极小（几乎重合）的情况
            zero_dist_mask = distances < epsilon
            if np.any(zero_dist_mask):
                z.ravel()[i] = np.mean(values[zero_dist_mask])
                continue
            
            # 计算权重
            weights = 1.0 / (distances ** power)
            
            # 计算加权平均值
            weighted_sum = np.sum(weights * values)
            sum_of_weights = np.sum(weights)
            
            if sum_of_weights > 0:
                z.ravel()[i] = weighted_sum / sum_of_weights
        
        return z

    def check_normality_and_transform(self, values, alpha=0.05):
        """检查正态性，如果不符合进行 Box-Cox 变换
        返回: transformed_values, lmbda, was_transformed, shift, shapiro_p
        """
        values = np.array(values, dtype=float)
        # shift 以保证正数用于 boxcox
        min_val = np.min(values)
        shift = 0.0
        if min_val <= 0:
            shift = abs(min_val) + 1e-6
            values_shifted = values + shift
        else:
            values_shifted = values.copy()

        # Shapiro-Wilk 检验（try 防止异常）
        try:
            _, p = stats.shapiro(values_shifted)
        except Exception:
            p = 1.0

        if p < alpha:
            # 尝试 Box-Cox（若全为正）
            try:
                transformed_values, lmbda = stats.boxcox(values_shifted)
                return transformed_values, lmbda, True, shift, p
            except Exception:
                # 退化到 log1p 当所有值> -1
                try:
                    if np.all(values_shifted > 0):
                        transformed_values = np.log1p(values_shifted)
                        return transformed_values, None, True, shift, p
                except Exception:
                    pass
                # 如果不能变换则返回原始
                return values, None, False, 0.0, p
        else:
            return values, None, False, 0.0, p

    def inverse_boxcox(self, y, lmbda):
        """Box-Cox 反变换"""
        if lmbda is None:
            return y
        y = np.array(y)
        if lmbda == 0:
            return np.exp(y)
        safe_y = y * lmbda + 1
        safe_y = np.where(safe_y <= 0, np.nan, safe_y)
        return np.power(safe_y, 1.0 / lmbda)

    def standardize(self, values):
        """标准化（返回 standardized, mean, std）"""
        mean = float(np.mean(values))
        std = float(np.std(values))
        std = std if std > 1e-10 else 1.0
        return (values - mean) / std, mean, std

    def destandardize(self, z, mean, std):
        """反标准化"""
        return z * std + mean

    def generate_grid(self, lons, lats, target_res=0.03, extent=None,
                      margin=0.5, margin_ratio=None, force_extent=None, clip_file="scBasin.geojson"):
        # 保证 target_res 合法
        if target_res is None or target_res <= 0:
            target_res = 0.03

        if force_extent is None and clip_file is not None:
            try:
                import geopandas as gpd
                basin = gpd.read_file(clip_file)
                minx, miny, maxx, maxy = basin.total_bounds
                force_extent = (minx, maxx, miny, maxy)
            except Exception:
                force_extent = None

        if force_extent is not None:
            minx, maxx, miny, maxy = force_extent
        elif extent is None:
            minx, maxx = float(np.min(lons)), float(np.max(lons))
            miny, maxy = float(np.min(lats)), float(np.max(lats))
        else:
            minx, maxx, miny, maxy = extent

        if force_extent is None:
            if margin_ratio is not None:
                dx = (maxx - minx) * margin_ratio
                dy = (maxy - miny) * margin_ratio
            else:
                dx = dy = margin
            minx -= dx; maxx += dx
            miny -= dy; maxy += dy

        nx = max(2, int((maxx - minx) / target_res) + 1)
        ny = max(2, int((maxy - miny) / target_res) + 1)
        grid_lon = np.linspace(minx, maxx, nx)
        grid_lat = np.linspace(miny, maxy, ny)
        grid_x, grid_y = np.meshgrid(grid_lon, grid_lat)
        return (grid_x, grid_y), grid_lon, grid_lat

    # ================== 半变异函数优化 ==================
    def select_best_model(self, lons, lats, values, candidate_models=None, *args, **kwargs):
        """
        自动选择最优半变异函数模型（支持传入多余参数，不会报错）
        说明：
          - scikit-gstat 并不内置 'linear'，遇到 'linear' 会跳过拟合（标记为 skipped）
          - 如果所有 skgstat 模型拟合失败且 'linear' 在候选中，则回退使用 'linear'
        """
        if candidate_models is None:
            candidate_models = ['spherical', 'exponential', 'gaussian']

        coords = np.column_stack((lons, lats))
        values = np.array(values, dtype=float)

        all_results = []
        best_model = None
        best_rmse = float("inf")

        print(f"[INFO] 共 {len(candidate_models)} 个候选半变异模型待评估...")

        # 记录是否包含 linear（可能用于回退）
        contains_linear = any(m.lower() == "linear" for m in candidate_models)

        for model_name in candidate_models:
            mn = model_name.lower()
            if mn == "linear":
                # scikit-gstat 默认不支持 linear variogram — 跳过拟合，但记录
                all_results.append({
                    "model": "linear",
                    "rmse": None,
                    "params": None,
                    "note": "skipped_by_skgstat (use PyKrige linear as fallback)"
                })
                print("[SKIP] scikit-gstat 不支持 'linear' 半变异模型，已跳过（将由 PyKrige 支持）。")
                continue

            try:
                V = Variogram(coords, values, model=mn, normalize=False)
                V.fit(method='trf')

                exp_lags = getattr(V, "bins", np.array([]))
                exp_gamma = getattr(V, "experimental", np.array([]))
                fit_gamma = V.fitted_model(exp_lags) if len(exp_lags) > 0 else np.array([])

                rmse = float(np.sqrt(np.mean((exp_gamma - fit_gamma) ** 2))) if len(exp_gamma) > 0 else float("inf")

                params = getattr(V, "parameters", None)
                if params is None or len(params) < 3:
                    nugget, sill, rng = 0.0, float(np.var(values)), float((np.max(lons) - np.min(lons)) / 3.0)
                else:
                    nugget, sill, rng = float(params[0]), float(params[1]), float(params[2])

                all_results.append({
                    "model": mn,
                    "rmse": rmse,
                    "params": {"nugget": nugget, "sill": sill, "range": rng},
                })

                print(f"[MODEL] {mn:<12s} → RMSE={rmse:.6f}, nugget={nugget:.3f}, sill={sill:.3f}, range={rng:.3f}")

                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model = mn

            except Exception as e:
                all_results.append({
                    "model": mn,
                    "error": str(e),
                    "rmse": None,
                    "params": None
                })
                print(f"[WARN] 模型 {mn} 拟合失败: {e}")
                continue

        # 回退逻辑：若没有找到任何 skgstat 模型，但候选里含 linear，则使用 linear（由 PyKrige 支持）
        if best_model is None and contains_linear:
            best_model = "linear"
            print("[WARN] 所有 scikit-gstat 候选拟合失败，回退至 'linear'（由 PyKrige 插值支持）")

        if best_model is None:
            print("[ERROR] 所有候选半变异模型拟合均失败（包括无可用回退）。")
        else:
            print(f"[INFO] ✅ 最优半变异模型: {best_model} (RMSE={best_rmse:.6f} if computed)")

        return best_model, all_results

    # ================== 主插值函数（按需求流程） ==================
    def interpolate(self, lons, lats, values, params: Dict[str, Any], target_res=0.03, extent=None):
        """
        完整流程：
          1) 自动选择方法（OK/UK）——除非 params 中已有 method
          2) 检验正态性并在需要时 Box-Cox 变换（记录变换信息）
          3) 标准化（用于变异函数拟合）
          4) 在候选变异模型中选择最优模型（skgstat）
          5) 使用 PyKrige（OK 或 UK）按选定变异模型进行插值，生成网格并返回
        返回包含：grid_x, grid_y, z, model_params, boxcox_info, variogram_candidates
        """
        try:
            # Step 0: 参数处理 & 验证输入
            lons = np.array(lons, dtype=float)
            lats = np.array(lats, dtype=float)
            values = np.array(values, dtype=float)
            mask = np.isfinite(values)
            lons, lats, values = lons[mask], lats[mask], values[mask]
            if len(values) == 0:
                return {"error": "❌ 无有效数据点，无法插值"}

            # Step 1: 自动选择方法（若 params 未指定）
            method_param = params.get("method")
            if not method_param or str(method_param).strip().lower() in ["none", "", "auto"]:
                # 自动判断使用 OK, UK 或 IDW
                method_info = self.suggest_kriging_method(lons, lats, values)
                kriging_method = method_info.get("suggestion", "ok").lower()
                method_reason = method_info.get("reason", "自动选择（默认 ok）")
            else:
                kriging_method = str(method_param).lower()
                method_reason = "用户指定"
            
            # Step 1.5: 如果是 IDW 或趋势面，则执行并提前返回
            if kriging_method in ['idw', 'trend']:
                (grid_x, grid_y), _, _ = self.generate_grid(lons, lats, target_res, extent)
                
                if kriging_method == 'idw':
                    power = float(params.get("idw_power", 2.0))
                    z = self._idw_interpolate(lons, lats, values, grid_x, grid_y, power=power)
                    model_specific_params = {"idw_power": power}
                else: # trend
                    degree = int(params.get("trend_degree", 2))
                    z = self._trend_surface_interpolate(lons, lats, values, grid_x, grid_y, degree=degree)
                    model_specific_params = {"trend_degree": degree}

                return {
                    "grid_x": grid_x.tolist(),
                    "grid_y": grid_y.tolist(),
                    "z": z.tolist(),
                    "model_params": {
                        "method": kriging_method,
                        "method_reason": method_reason,
                        **model_specific_params
                    },
                    "boxcox_info": {"was_transformed": False},
                    "variogram_candidates": []
                }

            # Step 2: 检验正态性并变换（如果需要）
            values_transformed, lmbda, was_transformed, shift, shapiro_p = self.check_normality_and_transform(values)
            # Step 3: 标准化（用于变异函数拟合）
            values_std, mean_val, std_val = self.standardize(values_transformed)

            # Step 4: 选择最优半变异模型（在标准化后的数据上）
            variogram_param = params.get("variogram_model")
            if not variogram_param or str(variogram_param).strip().lower() in ["none", "", "auto"]:
                candidate_models = params.get("candidate_variograms",
                                              ['spherical', 'exponential', 'gaussian'])
                best_model, variogram_candidates = self.select_best_model(
                    lons, lats, values_std, candidate_models=candidate_models
                )
                if not best_model:
                    best_model = "spherical"
            else:
                best_model = str(variogram_param).lower()
                variogram_candidates = [{"model": best_model, "note": "用户指定"}]

            # Step 5: 生成网格（注意：使用原始 lon/lat 范围）
            (grid_x, grid_y), grid_lon, grid_lat = self.generate_grid(lons, lats, target_res, extent)

            # Step 6: Kriging 插值（在标准化值空间执行）
            variogram_model_to_use = best_model
            variogram_parameters = None  # 让 PyKrige 自行拟合同名模型（更稳健）

            # 确定 drift_terms（仅在 UK 有效）
            drift_type = params.get("drift", "linear")
            drift_terms = None
            if kriging_method == "uk":
                if drift_type == "linear":
                    drift_terms = ["regional_linear"]
                elif drift_type == "quadratic":
                    drift_terms = ["regional_quadratic"]

            # PyKrige 支持 'linear' variogram（但 scikit-gstat 不一定支持）
            if kriging_method == "uk":
                uk = UniversalKriging(
                    lons, lats, values_std,
                    variogram_model=variogram_model_to_use,
                    variogram_parameters=variogram_parameters,
                    drift_terms=drift_terms,
                    verbose=False, enable_plotting=False
                )
                z_std, ss = uk.execute("grid", grid_lon, grid_lat, backend="vectorized")
                pykrige_fit = {
                    "model": uk.variogram_model,
                    "variogram_model_parameters": uk.variogram_model_parameters
                }
            else:
                ok = OrdinaryKriging(
                    lons, lats, values_std,
                    variogram_model=variogram_model_to_use,
                    variogram_parameters=variogram_parameters,
                    verbose=False, enable_plotting=False
                )
                z_std, ss = ok.execute("grid", grid_lon, grid_lat, backend="vectorized")
                pykrige_fit = {
                    "model": ok.variogram_model,
                    "variogram_model_parameters": ok.variogram_model_parameters
                }

            # Step 7: 反标准化与反变换（返回原始量纲）
            z_destd = self.destandardize(z_std, mean_val, std_val)
            if was_transformed:
                # 先做 Box-Cox 反变换（或 log 的反变换）
                if lmbda is not None:
                    z_back = self.inverse_boxcox(z_destd, lmbda)
                else:
                    # 如果使用 log1p 做的变换： inverse is expm1
                    z_back = np.expm1(z_destd)
                if shift != 0:
                    z_back = z_back - shift
            else:
                z_back = z_destd

            # Convert to python lists for JSON-serializable output
            return {
                "grid_x": grid_x.tolist(),
                "grid_y": grid_y.tolist(),
                "z": np.array(z_back).tolist(),
                "model_params": {
                    "method": kriging_method,
                    "method_reason": method_reason,
                    "selected_variogram": variogram_model_to_use,
                    "pykrige_fit": pykrige_fit,
                    "pretransform_stats": {"shapiro_p": shapiro_p, "was_transformed": bool(was_transformed)}
                },
                "boxcox_info": {"was_transformed": bool(was_transformed), "lambda": lmbda, "shift": shift},
                "variogram_candidates": variogram_candidates
            }

        except Exception as e:
            traceback.print_exc()
            return {"error": str(e)}

    # ================== 多层叠加插值 ==================
    def interpolate_and_overlay(self, datasets: Dict[str, Dict[str, List[float]]],
                                weights: Dict[str, float],
                                common_params: Dict[str, Any]):
        """
        对多个数据集进行插值，然后进行加权叠加。
        `datasets`: {'泥岩': {'lons': [...], 'lats': [...], 'values': [...]}, ...}
        `weights`: {'泥岩': 0.6, '灰岩': 0.4}
        `common_params`: 共享的插值参数
        """
        all_lons = np.concatenate([d['lons'] for d in datasets.values()])
        all_lats = np.concatenate([d['lats'] for d in datasets.values()])

        # 1. 生成一个能覆盖所有数据点的公共网格
        (grid_x, grid_y), grid_lon, grid_lat = self.generate_grid(all_lons, all_lats, common_params.get("target_res", 0.03))

        # 2. 对每个数据集进行插值
        individual_results = {}
        aligned_grids = {}
        for name, data in datasets.items():
            print(f"--- 开始插值: {name} ---")
            result = self.interpolate(
                lons=data['lons'],
                lats=data['lats'],
                values=data['values'],
                params=common_params
            )
            if "error" in result:
                return {"error": f"插值失败 at {name}: {result['error']}"}
            
            individual_results[name] = result
            
            # 3. 重采样到公共网格（如果需要）
            z_matrix = np.array(result['z'])
            if z_matrix.shape != grid_x.shape:
                # 这里需要一个重采样/插值函数，暂时用简单的 griddata
                from scipy.interpolate import griddata
                points = (np.array(result['grid_x']).flatten(), np.array(result['grid_y']).flatten())
                values = z_matrix.flatten()
                aligned_z = griddata(points, values, (grid_x, grid_y), method='cubic')
                aligned_grids[name] = np.nan_to_num(aligned_z)
            else:
                aligned_grids[name] = z_matrix

        # 4. 执行加权叠加
        final_grid = self._weighted_overlay(aligned_grids, weights)

        return {
            "individual_results": individual_results,
            "aligned_grids": {name: g.tolist() for name, g in aligned_grids.items()},
            "overlay_result": {
                "grid_x": grid_x.tolist(),
                "grid_y": grid_y.tolist(),
                "z": final_grid.tolist()
            }
        }

    def _weighted_overlay(self, grids: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
        """
        对齐后的栅格进行加权叠加
        """
        if not grids:
            raise ValueError("Grids dictionary is empty.")
        
        # 检查权重键是否与 grids 键匹配
        if set(grids.keys()) != set(weights.keys()):
            raise ValueError("Mismatch between grid names and weight names.")

        # 检查所有 grid 是否具有相同的形状
        first_grid_shape = next(iter(grids.values())).shape
        if not all(g.shape == first_grid_shape for g in grids.values()):
            raise ValueError("All grids must have the same shape for overlay.")

        # 归一化权重
        total_weight = sum(weights.values())
        if total_weight == 0:
            raise ValueError("Total weight cannot be zero.")
        normalized_weights = {name: w / total_weight for name, w in weights.items()}

        # 计算加权平均
        weighted_sum = np.zeros(first_grid_shape, dtype=float)
        for name, grid in grids.items():
            weighted_sum += grid * normalized_weights[name]
            
        return weighted_sum

    # ================== 交叉验证（保留） ==================
    def cross_validate(self, lons, lats, values, params):
        coords = np.column_stack((lons, lats))
        values = np.array(values, dtype=float)
        mask = np.isfinite(values)
        coords, values = coords[mask], values[mask]
        n_points = len(values)
        if n_points < 3:
            return {"error": "样本点太少"}

        residuals, variances = [], []

        for i in range(n_points):
            coords_train = np.delete(coords, i, axis=0)
            values_train = np.delete(values, i)
            coord_test = coords[i]
            value_test = values[i]

            try:
                res = self.interpolate(
                    lons=coords_train[:, 0],
                    lats=coords_train[:, 1],
                    values=values_train,
                    params=params,
                    target_res=0.03,
                    extent=(coord_test[0] - 0.03, coord_test[0] + 0.03,
                            coord_test[1] - 0.03, coord_test[1] + 0.03)
                )
                if "error" in res:
                    continue

                z_grid = np.array(res["z"])
                pred = z_grid[0][0]
                var_pred = np.var(values_train)
                residuals.append(pred - value_test)
                variances.append(var_pred)
            except Exception:
                continue

        if len(residuals) == 0:
            return {"error": "交叉验证未成功计算任何残差"}

        residuals = np.array(residuals)
        variances = np.array(variances)
        variances_safe = np.where(variances <= 0, 1e-6, variances)
        KRME = float(np.mean(residuals))
        KRMSE = float(np.mean(residuals ** 2 / variances_safe))

        method = params.get("method", "ok").upper()
        variogram_model = params.get("variogram_model", "spherical").lower()
        print(f"[INFO] 模型={method}-{variogram_model} -> KRME={KRME:.4f}, KRMSE={KRMSE:.4f}")

        return {"KRME": KRME, "KRMSE": KRMSE, "method": method, "variogram_model": variogram_model}
