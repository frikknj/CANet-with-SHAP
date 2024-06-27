class Opt():
    dataroot = 'missdor/'
    gpu_ids = '0'
    checkpoints_dir = 'checkpoints/'
    isTrain = None
    name = 'HI'
    task = 'all'

    # network
    init_type = 'normal'
    init_gain = 0.02

    # dataset
    preprocess = 'none'
    num_threads = 0  # threads for loading data
    batch_size = 16
    display_winsize = 512
    # addition

    # visdom
    display_freq = 100
    display_ncols = 4
    display_id = 1
    display_server = "http://localhost"
    display_env = 'main'
    display_port = 8097
    update_html_freq = 1000
    print_freq = 1000

    # saving
    save_epoch_freq = 5
    continue_train = False
    epoch_count = 1


    # hype-parameters
    niter = 50  # of iter at starting learning rate
    niter_decay = 50  # of iter to linearly decay learning rate to zero
    beta1 = 0.5  # momentum term of adam
    lr = 0.0002
    lr_policy = 'linear'  # learning rate policy. [linear | step | plateau | cosine]
    lr_decay_iters = 50
    load_iter = 0
    verbose = True
    no_html = False
    save_by_iter = True