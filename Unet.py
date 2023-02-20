from IPython.display import display_html
display_html("<script>Jupyter.notebook.kernel.restart()</script>",raw=True)

num_of_runs = 5
for i in range(1,num_of_runs+1):

    import warnings
    warnings.simplefilter("ignore")

    import uNetMain

    
    in_channel = 1 
    first_out_channel = 32
    trn_folder = '../data/bucket2/tr'
    val_folder = '../data/bucket2/val'
    goldBinary_folder = '../data/goldsBinaryAll'
    lr=1.0
    patience = 40
    min_delta = 0.0
      
    model_name_pre = 'weights/allWeightsB2'
      
    if i==1:
        model_name = model_name_pre + '.pth'
    else:
        model_name = model_name_pre + '_' + str(i) + '.pth'


    uNetMain.callMain(1, first_out_channel, trn_folder, val_folder, goldBinary_folder, lr, patience, min_delta, model_name)

    from IPython.display import display_html
    display_html("<script>Jupyter.notebook.kernel.restart()</script>",raw=True)