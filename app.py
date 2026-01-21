import streamlit as st
import re
from Cable import Cable
from Upload_Data import process_csv
import pandas as pd
import numpy as np

import plotly.express as px


def bucket_reason(text: str) -> str:
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

## find an example of Open, short, and high res -- ive only found high res
def DCR_continuity_histograms(master_failures):
    #use error dataframe from cables depending on type and create 4 graphs, 
    #first one uses the "detail" column to extract the reason for failure (high open, short)
    #make a 3 bar- chart to bucket these into the categories and display

    df = master_failures.copy()
    df["Category"] = df["Detail"].apply(bucket_reason)


    cat_counts = (
        df["Category"]
        .value_counts()
        .rename_axis("Category")
        .reset_index(name="Count")
    )

    
    fig_overall = px.bar(
        cat_counts, 
        x="Category", 
        y="Count", 
        text="Count",
        title="Master Failure Categories"
    )
    fig_overall.update_traces(textposition="outside")
    
    stacked = (
        df.groupby(["Channel", "Category"])
        .size()
        .reset_index(name="Count")
    )

    fig_stacked = px.bar(
        stacked,
        x="Channel",
        y="Count",
        color="Category",
        barmode="stack",
        title="Master Failures by Channel (Stacked)"
    )

    return fig_overall, fig_stacked

    #2nd and 3rd will display all the channels and the number of failures on each one -- of each category


def generate_leakage_histograms(master_df):
    
    max_edge = 10000 
    bin_size = 200 
    threshold = 2000
    values_df = master_df.drop(columns=["Channel"], errors="ignore").copy()
    values_df = values_df.apply(pd.to_numeric, errors="coerce")

    
    col_max = values_df.max(axis=0, skipna=True)
    col_max = col_max.replace([np.inf, -np.inf], np.nan).dropna()

    all_vals = values_df.to_numpy().ravel()
    all_vals = all_vals[np.isfinite(all_vals)]
    all_vals = all_vals[~np.isnan(all_vals)]

    edges = np.arange(0, max_edge + bin_size, bin_size)
    labels = [f"{int(edges[i])}-{int(edges[i+1])}" for i in range(len(edges) - 1)]
    labels.append(f"{int(max_edge)}+")

    
    max_df = pd.DataFrame({"cable": col_max.index, "value": col_max.values})
    max_df["bin"] = pd.cut(max_df["value"], bins=list(edges) + [np.inf],
                           labels=labels, right=False)

    all_df = pd.DataFrame({"value": all_vals})
    all_df["bin"] = pd.cut(all_df["value"], bins=list(edges) + [np.inf],
                           labels=labels, right=False)



    fig_max = px.histogram(
        max_df,
        x="bin",
        category_orders={"bin": labels},
        title=f"Max Leakage per Cable (bin={int(bin_size)})",
        labels={"bin": "Leakage Range"}
    )
    fig_max.update_layout(xaxis_title="Maximum Leakage", template="plotly_white")


    fig_all = px.histogram(
        all_df,
        x="bin",
        category_orders={"bin": labels},
        title=f"All Leakage Values (bin={int(bin_size)})",
        labels={"bin": "Leakage Range"}
    )
    fig_all.update_layout(xaxis_title="Leakage", template="plotly_white")


    if threshold is not None:
        # Determine which categorical index to target:
        if threshold >= max_edge:
            vline_x = len(labels) - 1  # overflow bin
        else:
            vline_x = int(np.floor(threshold / bin_size))
        for fig in (fig_max, fig_all):
            fig.add_vline(
                x=vline_x,
                line_dash="dash",
                line_color="red",
                annotation_text=f"{int(threshold)}",
                annotation_position="top right",
            )

    return fig_max, fig_all

        
def create_cable(cable_length, serial_number):
    return Cable(cable_length, serial_number)
def has_data(cables, test_type: str) -> bool:
    """
    Return True if 'cables' has data for the given cable_type and test_type.
    Adjust logic based on your actual data structure.
    """
    if cables is None:
        return False

    for cable in cables.values():
        if(test_type == "1s"):
            if(cable.leakage_1s is not None):
                return True
        elif(test_type == "leakage"):
            if(cable.leakage is not None):
                return True
        elif(test_type == "DCR"):
            if(cable.resistance is not None):
                return True
        elif(test_type == "continuity"):
            if(cable.continuity is not None):
                return True
    return False

# bring the histogram stuff over from the other site for leakage 

# create a graph for DCR failures -- reason from the failure dataframe 
# same for continuity 
# download master csv for all forms of test 
# do we want like a max resistance section or anything like how we have max leakage 
# make new tabs for inv_resistance and inv_continuity
def build_master_dataframe(
    cables: dict,
    attr_name: str,
):
    
    """ Build a master DataFrame for the given cable type and data attribute.
    - Deduplicates each cable's DataFrame by taking the max per key (first column).
    - Merges all cables on the shared first column via outer join.
    - If duplicates remain, collapses them by taking column-wise max per key. """
    

    dfs = []

    for cable in cables.values():

        df = getattr(cable, attr_name, None)
        if df is None or df.empty:
            continue

        # Work with first two columns: [shared_key, measurement]
        tmp = df.iloc[:, :2].copy()
        shared_col = tmp.columns[0]
        meas_col   = tmp.columns[1]

        # Ensure measurement is numeric for max calculation
        tmp[meas_col] = pd.to_numeric(tmp[meas_col], errors="coerce")

        # 1) Deduplicate within this cable: max per key
        tmp = (
            tmp.groupby(shared_col, as_index=False)
               .agg({meas_col: "max"})
        )

        # 2) Rename measurement column to the cable's serial number
        tmp = tmp.rename(columns={meas_col: cable.serial_number})

        dfs.append(tmp)


    # Use the first DataFrame's first column name as the join key
    master_df = dfs[0].copy()
    key_col = master_df.columns[0]

    # 3) Merge all cables on shared key
    for df_i in dfs[1:]:
        # Align key column name if differs
        if df_i.columns[0] != key_col:
            df_i = df_i.rename(columns={df_i.columns[0]: key_col})
        master_df = master_df.merge(df_i, on=key_col, how="outer")

    # 4) If any duplicate keys exist post-merge, collapse by max per column
    # Convert all measurement columns to numeric for max; leave key untouched
    meas_cols = [c for c in master_df.columns if c != key_col]
    for c in meas_cols:
        master_df[c] = pd.to_numeric(master_df[c], errors="coerce")

    master_df = (
        master_df.groupby(key_col, as_index=False)
                 .max(numeric_only=True)
    )

    # 5) Sort by the shared key (numeric if possible)
    master_df[key_col] = pd.to_numeric(master_df[key_col], errors="ignore")
    master_df = master_df.sort_values(by=key_col)

    return master_df, None 



st.set_page_config(page_title="Tesla Tools", layout="wide")

st.title("Tesla Tools")
uploaded_files = st.file_uploader("Upload your CSV files", type="csv", accept_multiple_files=True)
cables = {}

pattern = re.compile(r"(?<![A-Za-z0-9])0[0-4][A-Za-z0-9]{8}(?![A-Za-z0-9])", re.IGNORECASE)
if uploaded_files:
    for uploaded_file in uploaded_files:
        serial_number = "Unknown"
            
        target_text = uploaded_file.name

        match = pattern.search(target_text)  # or uploaded_file.name
        if match:
            serial_number = match.group()
            if len(serial_number) >= 2:
                    first_two_digits = serial_number[:2]
                    second_digit = first_two_digits[1]          
                    if second_digit == "0":
                        continue
                    elif second_digit == "1":
                        continue
                    elif second_digit == "3":
                        cable_length = 11
                    elif second_digit == "4":
                        cable_length = 15

            if serial_number in cables:
                cable = cables[serial_number]
            else:
                cable = create_cable(cable_length, serial_number)
                cables[serial_number] = cable
            process_csv(cable, uploaded_file)
            
                
    Continuity_Tab, Inverse_Continuity_Tab, DCR_Tab, Inverse_DCR_Tab, Leakage_1s_Tab, Final_Leakage_Tab = st.tabs(["Continuity", "Inverse Continuity", "DCR", "Inverse DCR", "1s Leakage", "Final Leakage"])

    with Continuity_Tab:
        st.subheader("Continuity")
        st.divider()
        has_continuity = any(
            getattr(cable, "continuity", None) is not None
            and not getattr(cable.continuity, "empty", True)
            for cable in cables
        )
        if(has_continuity):
            continuity_master = build_master_dataframe(cables, "continuity")[0]
        #generate failure table 
        continuity_master_failures = []
        for cable in cables.values():
            fail_df = cable.continuity_errors
            if cable.continuity is not None:
                if fail_df is not None and not fail_df.empty:
                    continuity_master_failures.append(fail_df.copy())
                    
        if continuity_master_failures:
            master_df = pd.concat(continuity_master_failures, ignore_index=True)


            master_df["Category"] = master_df["Detail"].apply(bucket_reason)
            fig_overall, fig_stacked = DCR_continuity_histograms(master_df)

            st.subheader("Continuity Failure Summary")
            st.plotly_chart(fig_overall, use_container_width=True)
            st.plotly_chart(fig_stacked, use_container_width=True)

            st.divider()

        if "continuity_figs" not in st.session_state:
            st.session_state.continuity_figs = {}

        #generate master csv etc 




        COL_LAYOUT = [1, 1, 5, 1]
        st.subheader("Processed Cables")

        header_cols = st.columns(COL_LAYOUT)
        header_cols[0].markdown("**Serial Number**")
        header_cols[1].markdown("**Length (in)**")
        header_cols[2].markdown("**Heatmap**")
        header_cols[3].markdown("**Download CSV**")
        #find all the cables with continuity data and create the buttons for the heatmap data
        # generate the master dataframe so histogram can be made etc 

        continuity_master_failures = []  # list of DataFrames
        for cable in cables.values():
            cols = st.columns(COL_LAYOUT)
            has_df = (cable.continuity is not None)            

            fig_slot = cols[2].container()

            if(has_df):
                cols[0].markdown(cable.serial_number)
                cols[1].markdown(cable.length)
                fig_exists = cable.serial_number in st.session_state.continuity_figs
                if(not fig_exists):
                    if cols[2].button(
                        "Generate Heatmap",
                        key=f"continuity_heatmap_{cable.serial_number}",
                    ):
                        #continuity _ heatmap 
                        fig = cable.continuity_heatmap("forward")
                        st.session_state.continuity_figs[cable.serial_number] = fig
                else:
                    cols[2].markdown("<span style='color:gray'>Heatmap generated ✔</span>", unsafe_allow_html=True)
                if cols[3].button(
                    "download csv",
                    key=f"continuity_csv_{cable.serial_number}",
                ):
                    print("hello world")
                
        
        stored_fig = st.session_state.continuity_figs.get(cable.serial_number)
        if stored_fig is not None:
            fig_slot.plotly_chart(stored_fig, use_container_width=True, key = f"continuity_fig{cable.serial_number}")
    with Inverse_Continuity_Tab:
        st.subheader("Inverse Continuity")
        st.divider()
        has_inverse_continuity = any(
            getattr(cable, "inv_continuity", None) is not None
            and not getattr(cable.inv_continuity, "empty", True)
            for cable in cables
        )
        if(has_inverse_continuity):
            continuity_master = build_master_dataframe(cables, "inv_continuity")[0]
        inverse_continuity_master_failures = []
        for cable in cables.values():
            fail_df = cable.inv_continuity_errors
            if cable.inv_continuity is not None:
                if fail_df is not None and not fail_df.empty:
                    inverse_continuity_master_failures.append(fail_df.copy())
                    
        if inverse_continuity_master_failures:
            master_df = pd.concat(inverse_continuity_master_failures, ignore_index=True)


            master_df["Category"] = master_df["Detail"].apply(bucket_reason)
            fig_overall, fig_stacked = DCR_continuity_histograms(master_df)

            st.subheader("inverse continuity Failure Summary")
            st.plotly_chart(fig_overall, use_container_width=True)
            st.plotly_chart(fig_stacked, use_container_width=True)

            st.divider()


        #generate failure table 
        if "inverse_continuity_figs" not in st.session_state:
            st.session_state.inverse_continuity_figs = {}
        #generate master csv etc 


        COL_LAYOUT = [1, 1, 5, 1]
        st.subheader("Processed Cables")

        header_cols = st.columns(COL_LAYOUT)
        header_cols[0].markdown("**Serial Number**")
        header_cols[1].markdown("**Length (in)**")
        header_cols[2].markdown("**Heatmap**")
        header_cols[3].markdown("**Download CSV**")
        #find all the cables with continuity data and create the buttons for the heatmap data
        # generate the master dataframe so histogram can be made etc 

        for cable in cables.values():
            cols = st.columns(COL_LAYOUT)
            fig_slot = cols[2].container()
            has_df = (cable.inv_continuity is not None)            

            if(has_df):      
                fig_exists = cable.serial_number in st.session_state.inverse_continuity_figs
                cols[0].markdown(cable.serial_number)
                cols[1].markdown(cable.length)
                if(not fig_exists):
                    if cols[2].button(
                        "Generate Heatmap",
                        key=f"inverse_continuity_heatmap_{cable.serial_number}",
                    ):
                        #continuity _ heatmap 
                        fig = cable.continuity_heatmap("inverse")
                        st.session_state.inverse_continuity_figs[cable.serial_number] = fig
                else:
                    cols[2].markdown("<span style='color:gray'>Heatmap generated ✔</span>", unsafe_allow_html=True)


                if cols[3].button(
                    "download csv",
                    key=f"inverse_continuity_csv_{cable.serial_number}",
                ):
                    print("hello world")
            
            stored_fig = st.session_state.inverse_continuity_figs.get(cable.serial_number)
            if stored_fig is not None:
                fig_slot.plotly_chart(stored_fig, use_container_width=True, key=f"inverse_continuity_fig{cable.serial_number}")
    with DCR_Tab:
        st.subheader("DC Resistance")
        has_DCR = any(
            getattr(cable, "resistance", None) is not None
            and not getattr(cable.resistance, "empty", True)
            for cable in cables
        )
        if(has_DCR):
            DCR_master = build_master_dataframe(cables, "resistance")[0]
        dcr_master_failures = []
        for cable in cables.values():
            fail_df = cable.resistance_errors
            if cable.resistance is not None:
                if fail_df is not None and not fail_df.empty:
                    dcr_master_failures.append(fail_df.copy())
                    
        if dcr_master_failures:
            master_df = pd.concat(dcr_master_failures, ignore_index=True)


            master_df["Category"] = master_df["Detail"].apply(bucket_reason)
            print(master_df)
            fig_overall, fig_stacked = DCR_continuity_histograms(master_df)

            st.subheader("DCR Failure Summary")
            st.plotly_chart(fig_overall, use_container_width=True)
            st.plotly_chart(fig_stacked, use_container_width=True)

            st.divider()


        #generate master csv 
        #create tables of failures 
        if "DCR_figs" not in st.session_state:
            st.session_state.DCR_figs = {}
        #create histogram of failure type


        COL_LAYOUT = [1, 1, 5, 1]
        st.subheader("Processed Cables")
        header_cols = st.columns(COL_LAYOUT)
        header_cols[0].markdown("**Serial Number**")
        header_cols[1].markdown("**Length (in)**")
        header_cols[2].markdown("**Heatmap**")
        header_cols[3].markdown("**Download CSV**")
        #find all the cables with continuity data and create the buttons for the heatmap data
        # generate the master dataframe so histogram can be made etc 
        for cable in cables.values():
            has_df = (cable.resistance is not None)            

            fig_slot = cols[2].container()

            if(has_df):
                fig_exists = cable.serial_number in st.session_state.DCR_figs
                cols[0].markdown(cable.serial_number)
                cols[1].markdown(cable.length)
                if(not fig_exists):
                    if cols[2].button(
                        "Generate Heatmap",
                        key=f"DCR_heatmap{cable.serial_number}",
                    ):
                        fig = cable.DCR_heatmap("forward")
                        st.session_state.DCR_figs[cable.serial_number] = fig

                else:
                    cols[2].markdown("<span style='color:gray'>Heatmap generated ✔</span>", unsafe_allow_html=True)

                if cols[3].button(
                    "download csv",
                    key=f"DCR_csv_{cable.serial_number}",
                ):
                    print("hello world")
            #find the reasons each board failed from the dataset 
            stored_fig = st.session_state.DCR_figs.get(cable.serial_number)
            if stored_fig is not None:
                fig_slot.plotly_chart(stored_fig, use_container_width=True,key=f"DCR_fig{cable.serial_number}")
    with Inverse_DCR_Tab:
        st.subheader("Inverse DC Resistance")
        has_inverse_DCR = any(
            getattr(cable, "inv_resistance", None) is not None
            and not getattr(cable.inv_resistance, "empty", True)
            for cable in cables
        )
        if(has_inverse_DCR):
            inverse_DCR_master = build_master_dataframe(cables, "inv_resistance")[0]
        #generate master csv 
        inverse_dcr_master_failures = []
        for cable in cables.values():
            fail_df = cable.inv_resistance_errors
            if cable.inv_resistance is not None:
                if fail_df is not None and not fail_df.empty:
                    inverse_dcr_master_failures.append(fail_df.copy())
                    
        if inverse_dcr_master_failures:
            master_df = pd.concat(inverse_dcr_master_failures, ignore_index=True)


            master_df["Category"] = master_df["Detail"].apply(bucket_reason)
            print(master_df)
            fig_overall, fig_stacked = DCR_continuity_histograms(master_df)

            st.subheader("Inverse DCR Failure Summary")
            st.plotly_chart(fig_overall, use_container_width=True)
            st.plotly_chart(fig_stacked, use_container_width=True)

            st.divider()
        if "inverse_DCR_figs" not in st.session_state:
            st.session_state.inverse_DCR_figs = {}
        #create tables of failures 
        #create histogram of failure type
        st.divider()

        COL_LAYOUT = [1, 1, 5, 1]
        st.subheader("Processed Cables")
        header_cols = st.columns(COL_LAYOUT)
        header_cols[0].markdown("**Serial Number**")
        header_cols[1].markdown("**Length (in)**")
        header_cols[2].markdown("**Heatmap**")
        header_cols[3].markdown("**Download CSV**")
        #find all the cables with continuity data and create the buttons for the heatmap data
        # generate the master dataframe so histogram can be made etc 
        inverse_dcr_master_failures = []
        for cable in cables.values():
            cols = st.columns(COL_LAYOUT)
            has_df = (cable.inv_resistance is not None)            
            fig_slot = cols[2].container()
            fig_exists = cable.serial_number in st.session_state.inverse_DCR_figs

            if(has_df):
                cols[0].markdown(cable.serial_number)
                cols[1].markdown(cable.length)
                if(not fig_exists):
                    if cols[2].button(
                        "Generate Heatmap",
                        key=f"inverse_DCR_heatmap{cable.serial_number}",
                    ):
                        fig = cable.DCR_heatmap("inverse")
                        st.session_state.inverse_DCR_figs[cable.serial_number] = fig
                else:
                    cols[2].markdown("<span style='color:gray'>Heatmap generated ✔</span>", unsafe_allow_html=True)



                if cols[3].button(
                    "download csv",
                    key=f"inverse_DCR_csv_{cable.serial_number}",
                ):
                    print("hello world")
            #find the reasons each board failed from the dataset 
            stored_fig = st.session_state.inverse_DCR_figs.get(cable.serial_number)
            if stored_fig is not None:
                fig_slot.plotly_chart(stored_fig, use_container_width=True, key=f"inverse_DCR_fig{cable.serial_number}")
    with Leakage_1s_Tab:
        leakage_1s_master_failures = []
        # add the "master" histograms from before
        st.subheader("1s Leakage")
        for cable in cables.values():
            fail_df = cable.leakage_1s_errors
            if cable.leakage_1s is not None:
                if fail_df is not None and not fail_df.empty:
                    leakage_1s_master_failures.append(fail_df.copy())
        gen_master = any(c.leakage_1s is not None and not c.leakage_1s.empty for c in cables.values())        
        if(gen_master):
            master_df = build_master_dataframe(cables, "leakage_1s")[0]
            print(master_df)
            fig_overall, fig_stacked = generate_leakage_histograms(master_df)

            st.subheader("1s leakage Summary")
            st.plotly_chart(fig_overall, use_container_width=True)
            st.plotly_chart(fig_stacked, use_container_width=True)

            st.divider()
        #leakage 1s heatmap and master dataframe    st.divider()
        COL_LAYOUT = [1, 1, 5, 1]
        st.subheader("Processed Cables")
        has_leakage_1s = any(
            getattr(cable, "leakage_1s", None) is not None
            and not getattr(cable.leakage_1s, "empty", True)
            for cable in cables
        )

        if "leakage_1s_figs" not in st.session_state:
            st.session_state.leakage_1s_figs = {}
        header_cols = st.columns(COL_LAYOUT)
        header_cols[0].markdown("**Serial Number**")
        header_cols[1].markdown("**Length (in)**")
        header_cols[2].markdown("**Heatmap**")
        header_cols[3].markdown("**Download CSV**")
        #find all the cables with continuity data and create the buttons for the heatmap data
        # generate the master dataframe so histogram can be made etc 
        leakage_1s_master_failures = []
        for cable in cables.values():
            cols = st.columns(COL_LAYOUT)
            has_df = (cable.leakage_1s is not None)            
            fig_slot = cols[2].container()

            if(has_df):
                cols[0].markdown(cable.serial_number)
                cols[1].markdown(cable.length)

                fig_exists = cable.serial_number in st.session_state.leakage_1s_figs
                if not fig_exists:
                    if cols[2].button(
                        "Generate Heatmap",
                        key=f"leakage_1s_heatmap_{cable.serial_number}",
                    ):
                        fig = cable.draw_heatmap_plotly(matrix_type = "leakage_1s")
                        st.session_state.leakage_1s_figs[cable.serial_number] = fig
                
                else:
                    cols[2].markdown("<span style='color:gray'>Heatmap generated ✔</span>", unsafe_allow_html=True)
        
                if cols[3].button(
                    "download csv",
                    key=f"leakage_1s_csv_{cable.serial_number}",
                ):
                    print("hello world")

                    #exit()
                    #continuity _ heatmap 
                    #display heatmap 
            
            stored_fig = st.session_state.leakage_1s_figs.get(cable.serial_number)
            if stored_fig is not None:
                fig_slot.plotly_chart(stored_fig, use_container_width=True, key=f"leakage_1s_fig{cable.serial_number}")

    with Final_Leakage_Tab:
        #add the "master" histograms from before using the same logic
        st.subheader("Final Leakage")
        st.divider()
        for cable in cables.values():
            fail_df = cable.leakage_1s_errors
            if cable.leakage_1s is not None:
                if fail_df is not None and not fail_df.empty:
                    leakage_1s_master_failures.append(fail_df.copy())
        gen_master = any(c.leakage is not None and not c.leakage.empty for c in cables.values())        
        if(gen_master):
            master_df = build_master_dataframe(cables, "leakage")[0]
            print(type(master_df))
            print(master_df)
            fig_overall, fig_stacked = generate_leakage_histograms(master_df)

            st.subheader("leakage Summary")
            st.plotly_chart(fig_overall, use_container_width=True)
            st.plotly_chart(fig_stacked, use_container_width=True)

            st.divider()

        COL_LAYOUT = [1, 1, 5, 1]
        st.subheader("Processed Cables")
        
        if "leakage_figs" not in st.session_state:
            st.session_state.leakage_figs = {}


        header_cols = st.columns(COL_LAYOUT)
        header_cols[0].markdown("**Serial Number**")
        header_cols[1].markdown("**Length (in)**")
        header_cols[2].markdown("**Heatmap**")
        header_cols[3].markdown("**Download CSV**")
        #find all the cables with continuity data and create the buttons for the heatmap data
        # generate the master dataframe so histogram can be made etc 
        final_leakage_failures = []
        for cable in cables.values():
            cols = st.columns(COL_LAYOUT)
            has_df = (cable.leakage is not None)            
  
            fig_slot = cols[2].container()

            if(has_df):
                fig_exists = cable.serial_number in st.session_state.leakage_figs

                cols[0].markdown(cable.serial_number)
                cols[1].markdown(cable.length)
                if(not fig_exists):
                    if cols[2].button(
                        "Generate Heatmap",
                        key=f"final_leakage_{cable.serial_number}",
                    ):
                        fig = cable.draw_heatmap_plotly(matrix_type = "leakage")
                        st.session_state.leakage_figs[cable.serial_number] = fig
                else:
                    cols[2].markdown("<span style='color:gray'>Heatmap generated ✔</span>", unsafe_allow_html=True)

                    
                if cols[3].button(
                    "download csv",
                    key=f"final_leakage_csv_{cable.serial_number}",
                ):
                    print("hello world")

                    #exit()
                    #continuity _ heatmap 
                    #display heatmap 

            stored_fig = st.session_state.leakage_figs.get(cable.serial_number)
            if stored_fig is not None:
                fig_slot.plotly_chart(stored_fig, use_container_width=True, key=f"final_leakage_fig{cable.serial_number}")