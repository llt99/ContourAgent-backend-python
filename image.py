import io
import os
import base64
import fiona
import numpy as np
import jenkspy
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from adjustText import adjust_text
from matplotlib.path import Path
from matplotlib.ticker import FuncFormatter
from shapely.geometry import shape, Polygon, MultiPolygon, mapping
from scipy.ndimage import gaussian_filter
import matplotlib.colors as mcolors
import matplotlib.patheffects as path_effects

matplotlib.rcParams['font.sans-serif'] = ['SimHei']


class MapRenderer:
    def lighten_cmap(self, cmap, factor=1.3):
        colors = cmap(np.linspace(0, 1, cmap.N))
        colors[:, :3] = np.clip(colors[:, :3] * (1 / factor) + (1 - 1 / factor), 0, 1)
        return mcolors.ListedColormap(colors)

    async def load_boundary_from_geojson(self, filepath="scBasin.geojson"):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"未找到 {filepath}")
        with fiona.open(filepath, "r", encoding="utf-8") as src:
            geoms = [shape(feat["geometry"]) for feat in src]
        if len(geoms) == 1:
            return geoms[0]
        return MultiPolygon([g for g in geoms if isinstance(g, Polygon)])

    async def mask_z_by_boundary(self, grid_x, grid_y, z, boundary_geom=None):
        """
        Ensure grid_x/grid_y are 2D meshgrids and mask z by boundary polygon(s).
        """
        boundary_geom = boundary_geom or await self.load_boundary_from_geojson()

        # Normalize grid_x, grid_y to 2D arrays (meshgrid)
        grid_x = np.array(grid_x)
        grid_y = np.array(grid_y)
        if grid_x.ndim == 1 and grid_y.ndim == 1:
            gx, gy = np.meshgrid(grid_x, grid_y)
        else:
            gx, gy = grid_x, grid_y

        z = np.array(z)
        if z.shape != gx.shape:
            # try to reshape if possible
            try:
                z = z.reshape(gx.shape)
            except Exception:
                raise ValueError("grid_x/grid_y and z shapes are incompatible")

        # Prepare polygons
        if isinstance(boundary_geom, Polygon):
            polys = [boundary_geom]
        elif isinstance(boundary_geom, MultiPolygon):
            polys = list(boundary_geom.geoms)
        else:
            polys = []

        paths = [Path(np.array(p.exterior.coords)) for p in polys if len(p.exterior.coords) >= 3]
        points = np.vstack((gx.ravel(), gy.ravel())).T
        mask = np.zeros(len(points), dtype=bool)
        for path in paths:
            mask |= path.contains_points(points)
        mask = mask.reshape(gx.shape)
        return np.where(mask, z, np.nan)

    def dms_formatter(self, x, pos=None, is_lat=False):
        deg = int(x)
        min_float = abs((x - deg) * 60)
        minute = int(min_float)
        second = int(round((min_float - minute) * 60))
        if second == 60:
            second = 0
            minute += 1
        if minute == 60:
            minute = 0
            deg += 1
        direction = 'N' if is_lat and x >= 0 else 'S' if is_lat else 'E' if x >= 0 else 'W'
        return f"{abs(deg)}°{minute}′ {direction}"

    def draw_north_arrow(self, ax, x=0.95, y=0.75, size=0.08):
        ax.annotate('N',
                    xy=(x, y), xytext=(x, y - size),
                    arrowprops=dict(facecolor='black', width=5, headwidth=15),
                    ha='center', va='center', fontsize=14,
                    xycoords=ax.transAxes)

    def compute_levels(self, z_masked, n_classes):
        """Compute levels consistently (jenks with fallback)."""
        z_flat = z_masked[~np.isnan(z_masked)]
        if len(z_flat) == 0:
            return np.linspace(0, 1, 6).tolist()
        if n_classes is None:
            n_classes = min(11, max(5, len(z_flat) // 10))
        n_classes = max(2, int(n_classes))
        try:
            levels = jenkspy.jenks_breaks(z_flat, n_classes)
            # jenks_breaks returns length = n_classes + 1 (breaks)
            return levels
        except Exception:
            return np.linspace(np.nanmin(z_flat), np.nanmax(z_flat), n_classes + 1).tolist()

    def normalize_levels(self, levels):
        """将等值线 levels 整化（四舍五入到合适间距）"""
        levels = np.array(levels, dtype=float)
        vmin, vmax = np.min(levels), np.max(levels)

        raw_interval = (vmax - vmin) / (len(levels) - 1)

        nice_steps = np.array([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100])
        step = nice_steps[np.argmin(np.abs(nice_steps - raw_interval))]

        new_min = np.floor(vmin / step) * step
        new_max = np.ceil(vmax / step) * step
        n_steps = int((new_max - new_min) / step) + 1

        nice_levels = np.linspace(new_min, new_max, n_steps)

        return nice_levels.tolist()

    async def _prepare_grid_and_levels(self, grid_x, grid_y, z, boundary_geom=None,
                                       smooth_sigma=0, n_classes=None, colormap="RdYlBu", lighten=False):
        """Return consistent gx, gy, z_masked, levels, cmap, norm for both image and vector outputs."""
        # Convert and ensure 2D meshgrid
        grid_x = np.array(grid_x)
        grid_y = np.array(grid_y)
        if grid_x.ndim == 1 and grid_y.ndim == 1:
            gx, gy = np.meshgrid(grid_x, grid_y)
        else:
            gx, gy = grid_x, grid_y

        # mask
        z_masked = await self.mask_z_by_boundary(gx, gy, z, boundary_geom)

        # smooth
        if smooth_sigma and smooth_sigma > 0:
            z_masked = gaussian_filter(z_masked, sigma=smooth_sigma)

        # Compute levels (consistent default)
        levels = self.compute_levels(z_masked, n_classes)

        # 颜色映射
        DIVERGING_CMAPS = ["RdYlBu", "RdYlGn", "bwr", "coolwarm", "seismic"]
        cmap_name = colormap
        if colormap in DIVERGING_CMAPS:
            cmap_name += "_r"
        base_cmap = plt.get_cmap(cmap_name)
        cmap = self.lighten_cmap(base_cmap, factor=1.3) if lighten else base_cmap
        norm = mcolors.BoundaryNorm(levels, cmap.N)

        return gx, gy, z_masked, levels, cmap, norm

    async def _render_map_image(self, grid_x, grid_y, z, first_result, points=None, boundary_geom=None,
                                task_text=None, variable="thickness", lithology=None,
                                smooth_sigma=0, n_classes=11, colormap="RdYlBu",
                                lighten=False, precomputed=None):
        """
        If 'precomputed' is provided, it must be a dict with keys
        gx, gy, z_masked, levels, cmap, norm. Otherwise this function will
        call _prepare_grid_and_levels() (backwards-compatible).
        """
        is_percentage = False

        if precomputed is None:
            gx, gy, z_masked, levels, cmap, norm = await self._prepare_grid_and_levels(
                grid_x, grid_y, z, boundary_geom, smooth_sigma=smooth_sigma,
                n_classes=n_classes, colormap=colormap, lighten=lighten
            )
        else:
            gx = precomputed["gx"]
            gy = precomputed["gy"]
            z_masked = precomputed["z_masked"]
            levels = precomputed["levels"]
            cmap = precomputed["cmap"]
            norm = precomputed["norm"]

        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

        # --- 区分处理分类数据和连续数据 ---
        if first_result and first_result.get("is_categorical"):
            categories = first_result["categories"]
            names = categories["names"]
            colors = categories["colors"]

            cmap_cat = mcolors.ListedColormap(colors)
            bounds = np.arange(len(names) + 1) - 0.5
            norm_cat = mcolors.BoundaryNorm(bounds, cmap_cat.N)

            # Use pcolormesh to avoid imshow pixel alignment issues
            pcm = ax.pcolormesh(gx, gy, z_masked, cmap=cmap_cat, norm=norm_cat, shading='auto')
            cbar = fig.colorbar(pcm, ax=ax, ticks=np.arange(len(names)), orientation="vertical", shrink=0.6, pad=0.03)
            cbar.set_ticklabels(names)
            cbar.set_label("沉积相分布", fontsize=12)
        else:
            # Continuous data
            cs = ax.contourf(gx, gy, z_masked, levels=levels, cmap=cmap, norm=norm)
            # Ensure colorbar ticks reflect levels (use level midpoints or boundaries)
            # Here use boundaries as ticks if not too many
            try:
                if len(levels) <= 12:
                    cbar = fig.colorbar(cs, ax=ax, ticks=levels, orientation="vertical", shrink=0.4, pad=0.03)
                else:
                    cbar = fig.colorbar(cs, ax=ax, orientation="vertical", shrink=0.4, pad=0.03)
            except Exception:
                cbar = fig.colorbar(cs, ax=ax, orientation="vertical", shrink=0.4, pad=0.03)

            is_percentage = variable.endswith("占比") or variable.lower() == "ratio"
            if variable == "地层厚度":
                cbar.set_label("地层厚度 (m)", fontsize=12)
            elif is_percentage:
                cbar.set_label(f"{lithology or variable} (%)", fontsize=12)
                cbar.ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
            else:
                cbar.set_label(variable, fontsize=12)

        # ----------------------------
        # 绘制钻井点
        # ----------------------------
        # if points:
        #     lons, lats = [], []
        #     texts = []
        #     y_offset = 0.02
        #
        #     for p in points:
        #         lon = (
        #             p.get("lon") or p.get("lng") or p.get("x") or
        #             p.get("longitude") or p.get("geo_X")
        #         )
        #         lat = (
        #             p.get("lat") or p.get("y") or p.get("latitude") or
        #             p.get("geo_Y")
        #         )
        #         name = (
        #             p.get("name") or p.get("well_name") or
        #             p.get("井名") or ""
        #         )
        #
        #         if lon is not None and lat is not None:
        #             try:
        #                 lon = float(lon)
        #                 lat = float(lat)
        #             except Exception:
        #                 continue
        #
        #             lons.append(lon)
        #             lats.append(lat)
        #
        #             # 绘制井名
        #             txt = ax.text(
        #                 lon,
        #                 lat + y_offset,
        #                 str(name),
        #                 fontsize=5,
        #                 ha="center",
        #                 va="bottom",
        #                 color="black",
        #                 zorder=6,
        #                 path_effects=[
        #                     path_effects.withStroke(linewidth=2, foreground="white")
        #                 ],
        #             )
        #             texts.append(txt)
        #
        #     if lons and lats:
        #         ax.scatter(
        #             lons, lats,
        #             c="red",
        #             s=20,
        #             edgecolors="white",
        #             linewidths=0.8,
        #             label="钻井点",
        #             zorder=5
        #         )
        #
        #     if texts:
        #         adjust_text(
        #             texts,
        #             only_move={'points': 'y', 'texts': 'y'},
        #             arrowprops=dict(arrowstyle="-", color="gray", lw=0.5)
        #         )

        # ----------------------------
        # 绘制边界
        # ----------------------------
        try:
            basin_geom = boundary_geom or await self.load_boundary_from_geojson()
            polys = [basin_geom] if isinstance(basin_geom, Polygon) else list(basin_geom.geoms)
            for poly in polys:
                x, y = poly.exterior.xy
                ax.plot(x, y, color="black", linewidth=1.2)
        except Exception as e:
            print("⚠️ 无法加载边界:", e)

        # ----------------------------
        # 坐标轴格式化
        # ----------------------------
        ax.xaxis.set_major_formatter(FuncFormatter(lambda val, pos: self.dms_formatter(val, pos, is_lat=False)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda val, pos: self.dms_formatter(val, pos, is_lat=True)))
        ax.set_xlim(102, 111)
        ax.set_ylim(27, 33)
        ax.set_aspect("equal", adjustable="box")

        # ----------------------------
        # 绘制北箭头
        # ----------------------------
        self.draw_north_arrow(ax, x=0.92, y=0.92, size=0.12)

        # ----------------------------
        # 设置标题
        # ----------------------------
        if task_text:
            title = task_text[2:].strip() if task_text.startswith("绘制") else task_text.strip()
        else:
            if is_percentage:
                title = f"{variable}图"
            elif variable == "地层厚度":
                title = f"地层厚度分布图"
            else:
                title = f"{variable}图"

        ax.set_title(title, fontsize=14, fontweight="bold")

        # ----------------------------
        # 输出为 base64
        # ----------------------------
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    async def generate_contours_polygons_geojson(
            self, grid_x, grid_y, z, first_result, n_classes=None, boundary_geom=None,
            colormap="RdYlBu", lighten=False, smooth_sigma=0, precomputed=None
    ):
        """
        If precomputed is provided (dict with gx,gy,z_masked,levels,cmap,norm), use it.
        Otherwise compute internally (backwards-compatible).
        """
        if precomputed is None:
            gx, gy, z_masked, levels, cmap, norm = await self._prepare_grid_and_levels(
                grid_x, grid_y, z, boundary_geom, smooth_sigma=smooth_sigma,
                n_classes=n_classes, colormap=colormap, lighten=lighten
            )
        else:
            gx = precomputed["gx"]
            gy = precomputed["gy"]
            z_masked = precomputed["z_masked"]
            levels = precomputed["levels"]
            cmap = precomputed["cmap"]
            norm = precomputed["norm"]

        # --- 区分处理分类数据和连续数据 ---
        if first_result and first_result.get("is_categorical"):
            categories = first_result["categories"]
            names = categories["names"]
            colors = categories["colors"]
            features = []
            for idx, name in enumerate(names):
                mask = (z_masked == idx)
                if not np.any(mask):
                    continue
                fig = plt.figure()
                cs = plt.contour(gx, gy, mask.astype(float), levels=[0.5])
                plt.close(fig)
                color = colors[idx]
                for segs in cs.allsegs:
                    for seg in segs:
                        if len(seg) < 3:
                            continue
                        poly = Polygon(seg)
                        if not poly.is_valid or poly.is_empty:
                            continue
                        features.append({
                            "type": "Feature",
                            "geometry": mapping(poly),
                            "properties": {"facies": name, "fill": color}
                        })
            return {
                "type": "FeatureCollection",
                "features": features,
                "properties": {"categories": names, "colors": colors}
            }
        else:
            # Continuous: use same levels/norm/cmap
            fig = plt.figure()
            cs = plt.contourf(gx, gy, z_masked, levels=levels, cmap=cmap, norm=norm)
            plt.close(fig)

            # Colors for each level (use norm on boundaries)
            colors_hex = [mcolors.to_hex(cmap(norm(level))) for level in levels[:-1]]
            features = []
            for i, segs in enumerate(cs.allsegs):
                level = float(cs.levels[i])
                color = colors_hex[i] if i < len(colors_hex) else None
                for seg in segs:
                    if len(seg) < 3:
                        continue
                    poly = Polygon(seg)
                    if not poly.is_valid or poly.is_empty:
                        continue
                    features.append({
                        "type": "Feature",
                        "geometry": mapping(poly),
                        "properties": {"value": level, "fill": color}
                    })
            return {
                "type": "FeatureCollection",
                "features": features,
                "properties": {
                    "contour_levels": [float(l) for l in levels],
                    "colors": colors_hex,
                    "colormap": colormap,
                    "lighten": lighten
                }
            }

    async def render_map(self, grid_x, grid_y, z, first_result, points=None, boundary_geom=None,
                         task_text=None, variable="thickness", lithology=None,
                         smooth_sigma=0, n_classes=None, colormap="RdYlBu",
                         lighten=False):
        """
        Central entry: prepare grid+levels once and pass precomputed to both image and geojson generators,
        ensuring image, vector and legend use identical levels and colormap.
        """
        gx, gy, z_masked, levels, cmap, norm = await self._prepare_grid_and_levels(
            grid_x, grid_y, z, boundary_geom, smooth_sigma=smooth_sigma,
            n_classes=n_classes, colormap=colormap, lighten=lighten
        )
        precomputed = {"gx": gx, "gy": gy, "z_masked": z_masked, "levels": levels, "cmap": cmap, "norm": norm}

        image_base64 = await self._render_map_image(
            grid_x, grid_y, z, first_result, points, boundary_geom,
            task_text, variable, lithology, smooth_sigma, n_classes, colormap, lighten,
            precomputed=precomputed
        )

        geojson = await self.generate_contours_polygons_geojson(
            grid_x, grid_y, z, first_result, n_classes, boundary_geom, colormap, lighten,
            smooth_sigma, precomputed=precomputed
        )

        # For client convenience, include levels and colors in response
        colors = geojson.get("properties", {}).get("colors", [])
        return {
            "image_base64": image_base64,
            "geojson": geojson,
            "colors": colors,
            "colormap": colormap,
            "lighten": lighten,
            "levels": levels
        }
