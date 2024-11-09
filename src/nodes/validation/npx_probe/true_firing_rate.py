import numpy as np
import pandas as pd
import copy

def get_log10fr_df(df_nv, df_ns, layers):
    
    # concat firing rates by layer
    fr_vivo = []
    fr_sili_sp = []
    layer_vivo = []
    layer_sili_sp = []
    
    # plot each layer in a panel
    for _, layer in enumerate(layers):

        # vivo
        fr_vivo_i = (
            df_nv["firing_rate"][df_nv["layer"] == layer].values.astype(np.float32).tolist()
        )
        fr_vivo += fr_vivo_i
        layer_vivo += [layer] * len(fr_vivo_i)

        # silico spontaneois
        fr_sili_sp_i = (
            df_ns["firing_rate"][df_ns["layer"] == layer].values.astype(np.float32).tolist()
        )
        fr_sili_sp += fr_sili_sp_i
        layer_sili_sp += [layer] * len(fr_sili_sp_i)

    # build plot dataset
    vivo_data = pd.DataFrame(data=np.array(fr_vivo), columns=["firing rate"])
    vivo_data["experiment"] = "M"
    vivo_data["layer"] = layer_vivo
    sili_data_sp = pd.DataFrame(data=np.array(fr_sili_sp), columns=["firing rate"])
    sili_data_sp["experiment"] = "NS"
    sili_data_sp["layer"] = layer_sili_sp
    plot_data = pd.concat([vivo_data, sili_data_sp], ignore_index=True)

    # drop sites outside layers
    mask = np.isin(plot_data["layer"], layers)
    plot_data = plot_data[mask]
    plot_data = plot_data.sort_values(by=["layer"])

    # we plot the stats over log10(firing rate) which reflects
    # bestwhat we see from the distribution plots (stats over raw data
    # is not visible). Note: the log of the median is the median of the log
    df = copy.copy(plot_data)
    df["firing rate"] = np.log10(df["firing rate"])
    return df