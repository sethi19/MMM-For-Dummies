import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation
from pymc_marketing.prior import Prior
from pymc_marketing.mmm.evaluation import compute_summary_metrics

# Streamlit app title
st.title("📊 Marketing Mix Modeling (MMM) with PyMC")

# File uploader
uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read CSV
    df = pd.read_csv(uploaded_file)
    
    # Display dataset preview
    st.write("### Preview of Uploaded CSV:")
    st.dataframe(df.head())

    # Select the Date column
    date_col = st.selectbox("📅 Select the Date Column", df.columns)
    
    # Convert to datetime
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Select target variable (y)
    y_col = st.selectbox("🎯 Select the Dependent Variable (y)", df.columns)

    # Select spend columns
    spend_cols = st.multiselect("💰 Select Media Spend Columns", df.columns)

    # Select control columns
    control_cols = st.multiselect("🛠️ Select Control Columns", df.columns)

    if st.button("Run MMM Model"):
        # Ensure required columns are selected
        if not spend_cols:
            st.warning("⚠️ Please select at least one media spend column.")
        elif not y_col:
            st.warning("⚠️ Please select the dependent variable.")
        else:
            # Prepare Data
            raw_data = df[[date_col, y_col] + spend_cols + control_cols].copy()
            raw_data.rename(columns={date_col: "date"}, inplace=True)

            # Compute total spend per channel
            total_spend_per_channel = raw_data[spend_cols].sum(axis=0).round(2)
            spend_share = total_spend_per_channel / total_spend_per_channel.sum().round(2)
            prior_sigma = spend_share.loc[spend_cols].to_numpy()

            # Prepare data for MMM
            X = raw_data[spend_cols + control_cols + ["date"]]  # ✅ FIXED
            y = raw_data[y_col].squeeze()  # ✅ FIXED: Ensures y is 1D

            # Define model configuration
            model_config = {
                "intercept": Prior("HalfNormal", sigma=0.2),
                "saturation_beta": Prior("HalfNormal", sigma=prior_sigma, dims="channel"),
                "saturation_lam": Prior("Gamma", alpha=3, beta=1, dims="channel"),
                "gamma_control": Prior("Normal", mu=0, sigma=0.3),
                "gamma_fourier": Prior("Laplace", mu=0, b=1),
                "likelihood": Prior("Normal", sigma=Prior("HalfNormal", sigma=0.5)),
            }

            sampler_config = {
                "progressbar": True,
                "chains": 2,
                "draws": 1000,
                "tune": 500,
            }

            # Instantiate MMM model
            mmm = MMM(
                model_config=model_config,
                sampler_config=sampler_config,
                date_column="date",  # ✅ FIXED
                adstock=GeometricAdstock(l_max=10),
                saturation=LogisticSaturation(),
                channel_columns=spend_cols,
                control_columns=control_cols,
                yearly_seasonality=1,
            )

            # Fit the model
            st.write("🔄 Running PyMC MMM Model...")
            mmm.fit(X=X, y=y, target_accept=0.95, random_seed=67)
            st.success("✅ MMM Model Training Complete!")

            # Sample from posterior predictive
            st.write("🔍 Generating Posterior Predictive Samples...")
            posterior_preds = mmm.sample_posterior_predictive(X, random_seed=45)

            # Compute R²
            summary_metrics = compute_summary_metrics(
                y_true=mmm.y,  # ✅ FIXED: No need for .to_numpy()
                y_pred=posterior_preds.y.values,  
                metrics_to_calculate=["r_squared"]  
            )

            # Extract R² value
            r_squared = summary_metrics["r_squared"]["mean"]

            # Display R² in Streamlit
            st.write("### Model Performance")
            st.metric(label="📈 R² Score", value=f"{r_squared:.4f}")

            # Plot Posterior Predictive
            fig = mmm.plot_posterior_predictive(original_scale=True)
            fig.gca().set(xlabel="Date", ylabel=y_col)

            # Display the plot in Streamlit
            st.pyplot(fig)

            # Plot Waterfall Components Decomposition
            fig1 = mmm.plot_waterfall_components_decomposition()
            st.write("### 📊 Media & Baseline Contributions")
            st.pyplot(fig1)

            ### --- Spend vs Effect DataFrame ---
            # Spend DF
            spend_df = pd.DataFrame(spend_share).reset_index()
            spend_df.columns = ["Channel", "Spend Share"]
            spend_df["spend_share"] = (spend_df["Spend Share"] * 100 ).round(2)
            spend_df = spend_df[['Channel', 'spend_share']]

            # Effect DF
            is_waterfall_data = mmm.compute_mean_contributions_over_time(original_scale=True)
            is_processed_waterfall_data = mmm._process_decomposition_components(data=is_waterfall_data)
            is_processed_waterfall_data.columns = ["Channel", "contribution", "percentage"]
            effect_df = is_processed_waterfall_data.copy()
            effect_df = effect_df[effect_df["Channel"].isin(spend_cols)]
            effect_df["effect_share"] = ((effect_df["contribution"] / effect_df["contribution"].sum()) * 100).round(2)
            effect_df = effect_df[['Channel', 'effect_share']]

            # Merge Spend and Effect DataFrames
            spend_effect = spend_df.merge(effect_df, on="Channel")
            spend_effect.sort_values(by="spend_share", ascending=True, inplace=True)

            # Display Spend-Effect DataFrame in Streamlit
            st.write("### 💰 Spend Share vs Effect Share")
            st.dataframe(spend_effect)

            # Spend vs. Effect Bar Chart
            x = np.arange(len(spend_effect["Channel"]))
            bar_width = 0.4  
            fig2, ax = plt.subplots(figsize=(10, 6))
            ax.barh(x - bar_width/2, spend_effect["spend_share"], height=bar_width, label="Spend Share", color="#9ec72c")
            ax.barh(x + bar_width/2, spend_effect["effect_share"], height=bar_width, label="Effect Share", color="black")
            ax.set_yticks(x)
            ax.set_yticklabels(spend_effect["Channel"])
            ax.set_xlabel("Share")
            ax.set_ylabel("Channel")
            ax.set_title("Spend Share vs Effect Share")
            ax.legend()

            # Display Bar Chart in Streamlit
            st.pyplot(fig2)

            # --- Direct Contribution Curves ---
            st.write("### 📈 Direct Contribution Curves")
            fig3 = mmm.plot_direct_contribution_curves()
            [ax.set(xlabel="x") for ax in fig3.axes]  # ✅ Label x-axis
            st.pyplot(fig3)

            model_path = "model.nc"
            mmm.save(model_path)

            # Option to download processed data
            with open(model_path, "rb") as f:
                st.download_button(
                    label="📥 Download Trained Model",
                    data=f,
                    file_name="model.nc",
                    mime="application/x-netcdf",
                )
