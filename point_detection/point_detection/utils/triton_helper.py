import tritonclient.http as httpclient
import sys
from tritonclient.utils import InferenceServerException


class TritonHelper():

    def __init__(self, url, verbose=False, concurrency=1) -> None:
        # Create triton client
        try:
            self.triton_client = httpclient.InferenceServerClient(url=url, verbose=verbose, concurrency=concurrency)
        except Exception as e:
            print("client creation failed: " + str(e))
            sys.exit(1)

    def set_in_out(self, inputs, outputs):
        self.inputs = []
        self.outputs = []
        for i in inputs:
            self.inputs.append(httpclient.InferInput(i['name'], i['shape'], i['precision']))
        for o in outputs:
            self.outputs.append(httpclient.InferRequestedOutput(o['name']))

    def call_server(self, data, input_name, model_name, request_id='1', model_version='1'):
        for i in self.inputs:
            if i.name() == input_name:
                i.set_data_from_numpy(data)
        # Inference
        try:
            response = self.triton_client.infer(model_name, inputs=self.inputs, request_id=request_id,
                                                model_version=model_version, outputs=self.outputs)
        except InferenceServerException as e:
            print("inference failed: " + str(e))
            sys.exit(1)
        return response
