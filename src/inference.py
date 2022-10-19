import torch
import torch.autograd as autograd

import time
import config
import logging
import numpy as np

from models import LSTMNvidiaNet

__all__ = ['inference']

def inference() -> None:
    
    # TODO: 2022-10-19 - Setup logger properly (seek for the right way & the right place)
    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)

    # Setup cuda
    device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device == 'cpu':
        logging.warning('CUDA device not found, migrating to CPU')

    logging.info('Device name: %s' % device)

    # Load model
    logging.info(f"Loading model: '{config.inference_model_path}'")
    model_state = torch.load(config.inference_model_path)
    logging.info(f"Model epoch {model_state.get('epoch')} loss: {model_state.get('eval_loss')}")

    model = LSTMNvidiaNet(num_blendshapes=config.n_blendshapes)
    model.load_state_dict(model_state.get('state_dict'))

    # Prepare loader
    features = torch.from_numpy(np.load(config.inference_features_path))
    target_dummy = torch.from_numpy(np.zeros(features.shape[0])) # NOTE: We don't care about labels here
    test_loader = DataLoader(TensorDataset(features, target_dummy), batch_size=batch_size, num_workers=2, shuffle=False)

    # Run inference
    model.eval()

    start_time = time.time()
    with torch.no_grad():
        for i, (input, _) in enumerate(test_loader):
            
            input_var = autograd.Variable(input.float())
            if torch.cuda.is_available():
                input_var = input_var.cuda()

            blendshapes = model(input_var)
            if i == 0:
                output = blendshapes.data
            else:
                output = torch.cat((output, blendshapes.data), 0)

    # Save inferred blendshapes
    output = output.cpu().numpy() * 100.0
    with open(config.inference_result_path, 'wb') as f:
        np.savetxt(f, output, fmt='%.6f')

    past_time = time.time() - start_time
    logging.info("Test finished in {:.4f} sec! Saved in {}".format(past_time, result_file))