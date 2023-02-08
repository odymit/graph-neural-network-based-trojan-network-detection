def test_activation_passing():
    def get_model():
        x = './shadow_model_ckpt/mnist/models5/shadow_jumbo_0.model'
        # load model 
        # Model = load_spec_model(father_model, '5')
        from model_lib.mnist_cnn_model import Model6 as Model
        model = Model(gpu=True)
        params = torch.load(x)
        model.load_state_dict(params)
        del params
        return model
    def get_graph():
        # load model detail 
        model_detail = {}
        model_detail_path = "./intermediate_data/model_detail.json"
        import json
        with open(model_detail_path, 'r') as f:
            model_detail = json.load(f)
        # print(model_detail)
        g = cnn2graph_activation(model, model_detail['mnist']['5'])
        dgl.save_graphs('./intermediate_data/grapj_test.bin', g)
        del model_detail
        return g 
    def get_image_dataset():
        # from utils_gnn import SGNACT
        GPU = True
        if GPU:
                torch.cuda.manual_seed_all(0)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                
        transform = transforms.Compose([
                    transforms.ToTensor(),
                ])
        BATCH_SIZE = 1
        # MNIST image dataset 
        trainset = torchvision.datasets.MNIST(root='./raw_data/', train=True, download=True, transform=transform)
        dataloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE)
        return dataloader
    model = get_model()
    g = get_graph()
    dataloader = get_image_dataset()
    for i, (x_in, y_in) in enumerate(dataloader):
        img = x_in
        label = y_in

        model_result, _ = model(img)
        passed_graph = activation_passing(img, g)
        graph_result = passed_graph.ndata['ft'][-1]
        _, n = passed_graph.ndata['ft_size'][-1]
        relu = torch.nn.functional.relu
        graph_result = relu(decode_fc_feat(graph_result, n))
        assert equals(model_result, graph_result), "output diffs: %d pic" % i
        if i > 100:
            break