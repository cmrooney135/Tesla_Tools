import re
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional
import pandas as pd

import seaborn as sns

class Cable:
    def __init__(self, length, serial_number):
        self.serial_number = serial_number
        self.length = length
        self.matrix: Optional[np.ndarray] = None
        self.leakage: Optional[pd.DataFrame] = None
        self.leakage_errors: Optional[pd.DataFrame] = None  
        self.leakage_1s: Optional[pd.DataFrame] = None
        self.leakage_1s_errors: Optional[pd.DataFrame] = None
        self.resistance: Optional[pd.DataFrame] = None
        self.resistance_errors: Optional[pd.DataFrame] = None
        self.inv_resistance: Optional[pd.DataFrame] = None
        self.inv_resistance_errors: Optional[pd.DataFrame] = None
        self.continuity: Optional[pd.DataFrame] = None
        self.continuity_errors: Optional[pd.DataFrame] = None
        self.inv_continuity: Optional[pd.DataFrame] = None
        self.inv_continuity_errors: Optional[pd.DataFrame] = None
    #region order of channels 
    #correct this
    Top = [
        "R64","F64","R63","F63","R62","F62","R61","F61",
        "R60","F60","R59","F59","R58","F58","R57","F57",
        "R56","F56","R55","F55","R54","F54","R53","F53",
        "R52","F52","R51","F51","R50","F50","R49","F49",
        
        "R48","F48","R47","F47","R46","F46","R45","F45",
        "R44","F44","R43","F43","R42","F42","R41","F41",
        "R40","F40","R39","F39","R38","F38","R37","F37",
        "R36","F36","R35","F35","R34","F34","R33","F33"

    ]



    TopS = [
        "RS64", "FS64", "RS63", "FS63",
        "RS62", "FS62", "RS61", "FS61",
        "RS60", "FS60", "RS59", "FS59",
        "RS58", "FS58", "RS57", "FS57",
        "RS56", "FS56", "RS55", "FS55",
        "RS54", "FS54", "RS53", "FS53",
        "RS52", "FS52", "RS51", "FS51",
        "RS50", "FS50", "RS49", "FS49",
        
        "RS48", "FS48", "RS47", "FS47",
        "RS46", "FS46", "RS45", "FS45",
        "RS44", "FS44", "RS43", "FS43",
        "RS42", "FS42", "RS41", "FS41",
        "RS40", "FS40", "RS39", "FS39",
        "RS38", "FS38", "RS37", "FS37",
        "RS36", "FS36", "RS35", "FS35",
        "RS34", "FS34", "RS33", "FS33"
    ]


    Bottom = [
        "F1","R1","F2","R2","F3","R3","F4","R4",
        "F5","R5","F6","R6","F7","R7","F8","R8",
        "F9","R9","F10","R10","F11","R11","F12","R12",
        "F13","R13","F14","R14","F15","R15","F16","R16",

        "F17","R17","F18","R18","F19","R19","F20","R20",
        "F21","R21","F22","R22","F23","R23","F24","R24",
        "F25","R25","F26","R26","F27","R27","F28","R28",
        "F29","R29","F30","R30","F31","R31","F32","R32"
    ]

    BottomS = [

        "FS1","RS1","FS2","RS2",
        "FS3","RS3","FS4","RS4",
        "RS5","FS5","FS6","RS6",
        "FS7","RS7","FS8","RS8",
        "FS9","RS9","FS10","RS10",
        "FS11","RS11","FS12","RS12",
        "FS13","RS13","FS14","RS14",
        "FS15","RS15","FS16","RS16",

        "FS17","RS17","FS18","RS18",
        "FS19","RS19","FS20","RS20",
        "FS21","RS21","FS22","RS22",
        "FS23","RS23","FS24","RS24",
        "FS25","RS25","FS26","RS26",
        "FS27","RS27","FS28","RS28",
        "FS29","RS29","FS30","RS30",
        "FS31","RS31","FS32","RS32"
    ]
    #endregion

    order = Top + TopS + BottomS + Bottom

    def extract_channel(*texts):
        
        pattern = re.compile(r"\((RS?\d+|FS?\d+)\)", re.IGNORECASE)

        for t in texts:
            if not isinstance(t, str) or not t:
                continue
            
            matches = pattern.findall(t)
            if matches:
                # Take the LAST one in case there are multiple
                return matches[-1].upper()

        return "0"

    
    def create_matrix(self, matrix_type): 
        if(matrix_type == "leakage"):
            df = self.leakage.copy()
            df_idx = (
            df.drop_duplicates(subset="Channel", keep="first")
            .set_index("Channel")
            )
            ordered = df_idx.reindex(self.order)
            ordered = ordered.reset_index()
            print(ordered)

            ordered["Leakage"] = ordered["Measured_pA"].fillna(0)
            ordered["Leakage"] = ordered["Leakage"].astype(float)
            
            return ordered
        elif(matrix_type == "leakage_1s"):
            df = self.leakage_1s.copy()
            df_idx = (
            df.drop_duplicates(subset="Channel", keep="first")
            .set_index("Channel")
            )
            ordered = df_idx.reindex(self.order)
            ordered = ordered.reset_index()
            ordered["Leakage"] = ordered["Measured_pA"].fillna(0)
            ordered["Leakage"] = ordered["Leakage"].astype(float)
            return ordered
        elif(matrix_type == "DCR"):
            df = self.resistance.copy()
            df_idx = (
            df.drop_duplicates(subset="Channel", keep="first")
            .set_index("Channel")
            )
            ordered = df_idx.reindex(self.order)
            ordered = ordered.reset_index()
            ordered["Resistance"] = ordered["Measured_R (mOhm)"].fillna(0)
            ordered["Resistance"] = ordered["Resistance"].astype(float)
            return ordered
        elif(matrix_type == "inverse_DCR"):
            df = self.inv_resistance.copy()
            df_idx = (
            df.drop_duplicates(subset="Channel", keep="first")
            .set_index("Channel")
            )
            ordered = df_idx.reindex(self.order)
            ordered = ordered.reset_index()
            ordered["Resistance"] = ordered["Measured_R (mOhm)"].fillna(0)
            ordered["Resistance"] = ordered["Resistance"].astype(float)
            return ordered
        elif(matrix_type == "continuity"):
            df = self.continuity.copy()
            df_idx = (
            df.drop_duplicates(subset="Channel", keep="first")
            .set_index("Channel")
            )
            ordered = df_idx.reindex(self.order)
            ordered = ordered.reset_index()
            ordered["Resistance"] = ordered["Measured_R (mOhm)"].fillna(0)
            ordered["Resistance"] = ordered["Resistance"].astype(float)
            return ordered
        elif(matrix_type == "inverse_continuity"):
            df = self.inv_continuity.copy()
            df_idx = (
            df.drop_duplicates(subset="Channel", keep="first")
            .set_index("Channel")
            )
            ordered = df_idx.reindex(self.order)
            ordered = ordered.reset_index()
            ordered["Resistance"] = ordered["Measured_R (mOhm)"].fillna(0)
            ordered["Resistance"] = ordered["Resistance"].astype(float)
            return ordered


    
        df_idx = (
            df.drop_duplicates(subset="Channel", keep="first")
            .set_index("Channel")
        )
        ordered = df_idx.reindex(self.order)
        ordered = ordered.reset_index()
        ordered["Leakage"] = ordered["Measured_pA"].fillna(0)
        ordered["Leakage"] = ordered["Leakage"].astype(float)
        return ordered
    def split_top_bottom(self, matrix_type):

        ordered = self.create_matrix(matrix_type)
        
        top_len     = len(self.Top)
        topS_len    = len(self.TopS)
        bottomS_len = len(self.BottomS)
        bottom_len  = len(self.Bottom)

        if(matrix_type == "continuity" or matrix_type =="inverse_continuity" or matrix_type == "DCR" or matrix_type =="inverse_DCR"):
            measured = ordered["Measured_R (mOhm)"].to_numpy()
            expected = ordered["Expected_R (mOhm)"].to_numpy()
        if(matrix_type == "leakage" or matrix_type == "leakage_1s"):
            measured = ordered["Measured_pA"].to_numpy()
            expected = ordered["Expected_pA"].to_numpy()
        diff_array = measured - expected   

        i0 = 0
        i1 = i0 + top_len
        i2 = i1 + topS_len
        i3 = i2 + bottomS_len
        i4 = i3 + bottom_len


        if i4 != len(diff_array):
            raise ValueError(
                f"Expected {i4} rows from channel lists, got {len(diff_array)}. "
                "Ensure ordered has exactly Top+TopS+BottomS+Bottom rows."
            )

        
        # Proper slices
        top     = diff_array[i0:i1]
        topS    = diff_array[i1:i2]
        bottomS = diff_array[i2:i3]
        bottom   = diff_array[i3:i4]

        return top, topS, bottomS, bottom
    

    def draw_heatmap_plotly(self, matrix_type):
        # Prepare data similarly to your current method
        top_leakage, topS_leakage, bottomS_leakage, bottom_leakage = self.split_top_bottom(matrix_type)

        # Reshape to a single row each (Plotly expects 2D arrays)
        top_leakage = top_leakage.reshape(1, -1)
        topS_leakage = topS_leakage.reshape(1, -1)
        bottomS_leakage = bottomS_leakage.reshape(1, -1)
        bottom_leakage = bottom_leakage.reshape(1, -1)

        # Convert your Matplotlib RGB tuples (0–1) to Plotly colorscale
        # Plotly colorscale expects list of (position, "rgb(r,g,b)") with r,g,b in 0–255
        colors = [
            (0, 0, 255),      # deep blue
            (77, 77, 255),    # intermediate blue
            (153, 153, 255),  # light blue
            (255, 255, 255),  # white
            (255, 153, 153),  # light red
            (255, 77, 77),    # intermediate red
            (255, 0, 0)       # full red
        ]
        nodes = np.linspace(0, 1, len(colors))
        custom_colorscale = [(float(p), f"rgb({r},{g},{b})") for p, (r, g, b) in zip(nodes, colors)]

        # Create 4-row subplot layout.
        # Height ratios roughly match your Matplotlib layout (1.0, 0.25, 0.25, 1.0).
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=False,
            vertical_spacing=0.2,
            row_heights=[0.38, 0.09, 0.09, 0.38],
        )

        # Common settings
        zmin, zmax = 0, 1500.0

        # Row 1: Top
        fig.add_trace(
            go.Heatmap(
                z=top_leakage,
                x=self.Top,
                y=[''],
                colorscale=custom_colorscale,
                zmin=zmin, zmax=zmax,
                colorbar=dict(title="Leakage (pA)"),
                showscale=True  # show once; you can set showscale=False on others
            ),
            row=1, col=1
        )

        # Row 2: TopS
        fig.add_trace(
            go.Heatmap(
                z=topS_leakage,
                x=self.TopS,
                y=[''],
                colorscale=custom_colorscale,
                zmin=zmin, zmax=zmax,
                showscale=False  # hide to avoid multiple bars
            ),
            row=2, col=1
        )

        # Row 3: BottomS
        fig.add_trace(
            go.Heatmap(
                z=bottomS_leakage,
                x=self.BottomS,
                y=[''],
                colorscale=custom_colorscale,
                zmin=zmin, zmax=zmax,
                showscale=False
            ),
            row=3, col=1
        )

        # Row 4: Bottom
        fig.add_trace(
            go.Heatmap(
                z=bottom_leakage,
                x=self.Bottom,
                y=[''],
                colorscale=custom_colorscale,
                zmin=zmin, zmax=zmax,
                showscale=False
            ),
            row=4, col=1
        )


        fig.update_xaxes(side="top",    row=1, col=1)  # Top
        fig.update_xaxes(side="top",    row=2, col=1)  # TopS
        fig.update_xaxes(side="bottom", row=3, col=1)  # BottomS
        fig.update_xaxes(side="bottom", row=4, col=1)  # Bottom


        # Rotate x tick labels to 90 degrees
        for r in [1, 2, 3, 4]:
            fig.update_xaxes(tickangle=90, row=r, col=1)

        # Remove y-axis title text (we have subplot titles already)
        for r in [1, 2, 3, 4]:
            fig.update_yaxes(title="", row=r, col=1)




        fig.update_layout(
            title={
                "text": f"Heatmap for cable with SN: {self.serial_number}",
                "x": 0.5,
                "xanchor": "center",
                "y": 0.98,
                "yanchor": "top",
                "pad": {"b": 20},   # space BELOW the title
                "font": {"size": 20},

            },
            margin=dict(l=40, r=40, t=90, b=70),  # more top margin
            height=400,
        )


        return fig
    
    def continuity_heatmap(self, type):
        if(type == "forward"):
            cable_type = "continuity"
        elif(type == "inverse"):
            cable_type = "inverse_continuity"
        top_continuity, topS_continuity, bottomS_continuity, bottom_continuity = self.split_top_bottom(cable_type)

        # Reshape to a single row each (Plotly expects 2D arrays)
        top_continuity = top_continuity.reshape(1, -1)
        topS_continuity = topS_continuity.reshape(1, -1)
        bottomS_continuity = bottomS_continuity.reshape(1, -1)
        bottom_continuity = bottom_continuity.reshape(1, -1)

        colors = [
            (0, 0, 255),      # deep blue
            (77, 77, 255),    # intermediate blue
            (153, 153, 255),  # light blue
            (255, 255, 255),  # white
            (255, 153, 153),  # light red
            (255, 77, 77),    # intermediate red
            (255, 0, 0)       # full red
        ]
        nodes = np.linspace(0, 1, len(colors))
        custom_colorscale = [(float(p), f"rgb({r},{g},{b})") for p, (r, g, b) in zip(nodes, colors)]

        # Create 4-row subplot layout.
        # Height ratios roughly match your Matplotlib layout (1.0, 0.25, 0.25, 1.0).
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=False,
            vertical_spacing=0.2,
            row_heights=[0.38, 0.09, 0.09, 0.38],
        )

        # Common settings
        zmin, zmax = -500, 500

        # Row 1: Top
        fig.add_trace(
            go.Heatmap(
                z=top_continuity,
                x=self.Top,
                y=[''],
                colorscale=custom_colorscale,
                zmin=zmin, zmax=zmax,
                colorbar=dict(title="continuity (mOhm)"),
                showscale=True  # show once; you can set showscale=False on others
            ),
            row=1, col=1
        )

        # Row 2: TopS
        fig.add_trace(
            go.Heatmap(
                z=topS_continuity,
                x=self.TopS,
                y=[''],
                colorscale=custom_colorscale,
                zmin=zmin, zmax=zmax,
                showscale=False  # hide to avoid multiple bars
            ),
            row=2, col=1
        )

        # Row 3: BottomS
        fig.add_trace(
            go.Heatmap(
                z=bottomS_continuity,
                x=self.BottomS,
                y=[''],
                colorscale=custom_colorscale,
                zmin=zmin, zmax=zmax,
                showscale=False
            ),
            row=3, col=1
        )

        # Row 4: Bottom
        fig.add_trace(
            go.Heatmap(
                z=bottom_continuity,
                x=self.Bottom,
                y=[''],
                colorscale=custom_colorscale,
                zmin=zmin, zmax=zmax,
                showscale=False
            ),
            row=4, col=1
        )


        fig.update_xaxes(side="top",    row=1, col=1)  # Top
        fig.update_xaxes(side="top",    row=2, col=1)  # TopS
        fig.update_xaxes(side="bottom", row=3, col=1)  # BottomS
        fig.update_xaxes(side="bottom", row=4, col=1)  # Bottom


        # Rotate x tick labels to 90 degrees
        for r in [1, 2, 3, 4]:
            fig.update_xaxes(tickangle=90, row=r, col=1)

        # Remove y-axis title text (we have subplot titles already)
        for r in [1, 2, 3, 4]:
            fig.update_yaxes(title="", row=r, col=1)




        fig.update_layout(
            title={
                "text": f"{type} continuity heatmap for cable with SN: {self.serial_number}",
                "x": 0.5,
                "xanchor": "center",
                "y": 0.98,
                "yanchor": "top",
                "pad": {"b": 20},   # space BELOW the title
                "font": {"size": 20},

            },
            margin=dict(l=40, r=40, t=90, b=70),  # more top margin
            height=400,
        )

        return fig
    #def split_top_bottom_Delta(self, type):

    def DCR_heatmap(self, type):
        if(type == "forward"):
            cable_type = "DCR"
        elif(type == "inverse"):
            cable_type = "inverse_DCR"
        top_resistance, topS_resistance, bottomS_resistance, bottom_resistance = self.split_top_bottom(cable_type)

        # Reshape to a single row each (Plotly expects 2D arrays)
        top_resistance = top_resistance.reshape(1, -1)
        topS_resistance = topS_resistance.reshape(1, -1)
        bottomS_resistance = bottomS_resistance.reshape(1, -1)
        bottom_resistance = bottom_resistance.reshape(1, -1)

        colors = [
            (0, 0, 255),      # deep blue
            (77, 77, 255),    # intermediate blue
            (153, 153, 255),  # light blue
            (255, 255, 255),  # white
            (255, 153, 153),  # light red
            (255, 77, 77),    # intermediate red
            (255, 0, 0)       # full red
        ]
        nodes = np.linspace(0, 1, len(colors))
        custom_colorscale = [(float(p), f"rgb({r},{g},{b})") for p, (r, g, b) in zip(nodes, colors)]

        # Create 4-row subplot layout.
        # Height ratios roughly match your Matplotlib layout (1.0, 0.25, 0.25, 1.0).
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=False,
            vertical_spacing=0.2,
            row_heights=[0.38, 0.09, 0.09, 0.38],
        )

        # Common settings
        zmin, zmax = -500, 500

        # Row 1: Top
        fig.add_trace(
            go.Heatmap(
                z=top_resistance,
                x=self.Top,
                y=[''],
                colorscale=custom_colorscale,
                zmin=zmin, zmax=zmax,
                colorbar=dict(title="resistance (mOhm)"),
                showscale=True  # show once; you can set showscale=False on others
            ),
            row=1, col=1
        )

        # Row 2: TopS
        fig.add_trace(
            go.Heatmap(
                z=topS_resistance,
                x=self.TopS,
                y=[''],
                colorscale=custom_colorscale,
                zmin=zmin, zmax=zmax,
                showscale=False  # hide to avoid multiple bars
            ),
            row=2, col=1
        )

        # Row 3: BottomS
        fig.add_trace(
            go.Heatmap(
                z=bottomS_resistance,
                x=self.BottomS,
                y=[''],
                colorscale=custom_colorscale,
                zmin=zmin, zmax=zmax,
                showscale=False
            ),
            row=3, col=1
        )

        # Row 4: Bottom
        fig.add_trace(
            go.Heatmap(
                z=bottom_resistance,
                x=self.Bottom,
                y=[''],
                colorscale=custom_colorscale,
                zmin=zmin, zmax=zmax,
                showscale=False
            ),
            row=4, col=1
        )


        fig.update_xaxes(side="top",    row=1, col=1)  # Top
        fig.update_xaxes(side="top",    row=2, col=1)  # TopS
        fig.update_xaxes(side="bottom", row=3, col=1)  # BottomS
        fig.update_xaxes(side="bottom", row=4, col=1)  # Bottom


        # Rotate x tick labels to 90 degrees
        for r in [1, 2, 3, 4]:
            fig.update_xaxes(tickangle=90, row=r, col=1)

        # Remove y-axis title text (we have subplot titles already)
        for r in [1, 2, 3, 4]:
            fig.update_yaxes(title="", row=r, col=1)




        fig.update_layout(
            title={
                "text": f"{type} resistance heatmap for cable with SN: {self.serial_number}",
                "x": 0.5,
                "xanchor": "center",
                "y": 0.98,
                "yanchor": "top",
                "pad": {"b": 20},   # space BELOW the title
                "font": {"size": 20},

            },
            margin=dict(l=40, r=40, t=90, b=70),  # more top margin
            height=400,
        )

        return fig
    
    def bucket_reason(self, text: str) -> str:
        t = text.lower()
        if "shorted" in t:
            return "Shorted"
        if "miswire" in t:
            return "Miswire"
        if "high" in t:
            return "High"
        if "open" in t:
            return "Open"
        if "missing" in t:
            return "Wire Missing"
        return "Other"
        
    
    def map_defects_to_order(self, error_df):
        # Assign a bucket category to each row (only where Detail exists)
        error_df["Category"] = error_df["Detail"].apply(
            lambda x: self.bucket_reason(x) if pd.notna(x) else ""
        )

        # Keep only unique channels (first occurrence)
        df_idx = (
            error_df.drop_duplicates(subset="Channel", keep="first")
            .set_index("Channel")
        )

        # Reindex to your fixed order — missing channels become NaN
        ordered = df_idx.reindex(self.order)
        
        ordered = ordered[["Category"]].reset_index()

        # Make Category blank instead of NaN
        ordered["Category"] = ordered["Category"].fillna("")

        print(ordered)

        return ordered


    def category_heatmap(self, categories_df):
        """
        Render a 4-row categorical heatmap (Top, TopS, BottomS, Bottom) where each cell
        shows the defect Category for that channel. Missing channels render as blank.
        Expects categories_df with columns ["Channel", "Category"].
        """

        # Map Channel -> Category (first occurrence wins)
        cat_map = (
            categories_df.drop_duplicates("Channel", keep="first")
            .set_index("Channel")["Category"]
            .to_dict()
        )

        # Build 1×N label rows aligned to each section; blank "" if not present
        top_labels     = np.array([cat_map.get(ch, "") for ch in self.Top], dtype=object).reshape(1, -1)
        topS_labels    = np.array([cat_map.get(ch, "") for ch in self.TopS], dtype=object).reshape(1, -1)
        bottomS_labels = np.array([cat_map.get(ch, "") for ch in self.BottomS], dtype=object).reshape(1, -1)
        bottom_labels  = np.array([cat_map.get(ch, "") for ch in self.Bottom], dtype=object).reshape(1, -1)

        # Determine categories → codes, with "" (blank) as 0
        cats = categories_df["Category"].dropna().unique().tolist()
        cats = [c for c in cats if c != ""]
        categories = [""] + cats
        cat_to_code = {c: i for i, c in enumerate(categories)}
        code_to_cat = {i: c for c, i in cat_to_code.items()}

        # Encode to integer codes
        enc = np.vectorize(lambda x: cat_to_code.get(x, 0))
        top_codes     = enc(top_labels).astype(int)
        topS_codes    = enc(topS_labels).astype(int)
        bottomS_codes = enc(bottomS_labels).astype(int)
        bottom_codes  = enc(bottom_labels).astype(int)

        # Build a discrete colorscale (first color is for blank/no defect)
        palette = [
            "#e6e6e6",  # "" blank
            "#1f77b4",  # cat 1
            "#ff7f0e",  # cat 2
            "#2ca02c",  # cat 3
            "#d62728",  # cat 4
            "#9467bd",  # (extra if more than 4)
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]
        if len(categories) > len(palette):
            raise ValueError(f"Add more colors to the palette (need {len(categories)}).")

        n = len(categories)
        cmin, cmax = -0.5, n - 0.5
        colorscale = []
        if n == 1:
            colorscale = [(0.0, palette[0]), (1.0, palette[0])]
        else:
            # flat steps: duplicate each stop so no gradient between codes
            for i in range(n):
                pos = i / (n - 1)
                colorscale.append((pos, palette[i]))
                colorscale.append((min(pos + 1e-6, 1.0), palette[i]))

        # Create the 4-row subplot
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=False,
            vertical_spacing=0.2,
            row_heights=[0.38, 0.09, 0.09, 0.38],
        )

        # Row 1: Top (show legend/colorbar here only)
        fig.add_trace(
            go.Heatmap(
                z=top_codes, x=self.Top, y=[""],
                zmin=cmin, zmax=cmax, colorscale=colorscale,
                colorbar=dict(
                    title="Category",
                    tickmode="array",
                    tickvals=list(range(n)),
                    ticktext=[code_to_cat[i] for i in range(n)],
                ),
                showscale=False,
            ),
            row=1, col=1
        )

        # Row 2: TopS
        fig.add_trace(
            go.Heatmap(
                z=topS_codes, x=self.TopS, y=[""],
                zmin=cmin, zmax=cmax, colorscale=colorscale,
                showscale=False,
            ),
            row=2, col=1
        )

        # Row 3: BottomS
        fig.add_trace(
            go.Heatmap(
                z=bottomS_codes, x=self.BottomS, y=[""],
                zmin=cmin, zmax=cmax, colorscale=colorscale,
                showscale=False,
            ),
            row=3, col=1
        )

        # Row 4: Bottom
        fig.add_trace(
            go.Heatmap(
                z=bottom_codes, x=self.Bottom, y=[""],
                zmin=cmin, zmax=cmax, colorscale=colorscale,
                showscale=False,
            ),
            row=4, col=1
        )

        # Axes: Top/TopS labels above; BottomS/Bottom below; rotate ticks
        fig.update_xaxes(side="top",    tickangle=90, row=1, col=1)
        fig.update_xaxes(side="top",    tickangle=90, row=2, col=1)
        fig.update_xaxes(side="bottom", tickangle=90, row=3, col=1)
        fig.update_xaxes(side="bottom", tickangle=90, row=4, col=1)
        for r in (1, 2, 3, 4):
            fig.update_yaxes(title="", row=r, col=1)

        fig.update_layout(
            title=f"Categorical defect map for SN: {self.serial_number}",
            margin=dict(l=40, r=40, t=90, b=70),
            height=400,
        )
        
        for i, cat in enumerate(categories):
            fig.add_trace(
                go.Scatter(
                    x=[None], y=[None],
                    mode="markers",
                    marker=dict(size=12, color=palette[i]),
                    legendgroup=cat,
                    showlegend=True,
                    name=cat if cat != "" else "No Defect"
                )
            )

        return fig



