import os
import requests
from uuid import uuid4

from requests import Response
from requests.exceptions import RequestException
from requests_toolbelt.multipart.encoder import MultipartEncoder
from typing import *
from dataclasses import dataclass, asdict


@dataclass
class RequestError:
    error: str

    def to_dict(self):
        return asdict(self)


class Service:
    def __init__(self):
        """Quasi-Singleton that is used to represent the BioCheck REST API and its methods therein."""
        self.endpoint_root = "https://biochecknet.biosimulations.org"
        root_response = self._test_root()
        print(root_response)

    def submit(self, omex_filepath: str, simulators: List[str], include_outputs: bool = True, comparison_id: str = None, ground_truth_report_path: Optional[str] = None) -> Union[Dict[str, str], RequestError]:
        """Submit a new uniform time course comparison job to the service and return confirmation of job submission.

            Args:
                omex_filepath:`str`: The path to the omex file to submit.
                simulators:`List[str]`: The list of simulators to include in comparison.
                include_outputs:`bool, optional`: Whether to include the output data used to calculate comparison in the job results on result fetch. Defaults to True.
                comparison_id:`str, optional`: The unique identifier for the comparison job. Defaults to None. If `None` is passed, a comparison id of `bio_check-request-<UUID>` is generated.
                ground_truth_report_path:`str, optional`: The path to the ground truth report file to include in comparison. Defaults to None.

            Returns:
                A dictionary containing the job submission results. **Note**: the return status should read `PENDING`.
        """
        endpoint = self._format_endpoint('utc-comparison')

        # configure params
        _id = comparison_id or "bio_check-request-" + str(uuid4())
        _report = ('ground_truth_file', open(ground_truth_report_path, 'rb'), 'application/octet-stream') if ground_truth_report_path else None
        multidata = MultipartEncoder(fields={
            'uploaded_file': ('omex_file', open(omex_filepath, 'rb'), 'application/octet-stream'),
            'simulators': ','.join(simulators),
            'include_outputs': str(include_outputs).lower(),
            'comparison_id': _id,
            'ground_truth_report': _report
        })
        headers = {'Content-Type': multidata.content_type}

        try:
            response = requests.post(endpoint, headers=headers, data=multidata)
            response.raise_for_status()
            self._check_response(response)
            return response.json()
        except Exception as e:
            return RequestError(error=str(e))

    def fetch(self, comparison_id: str) -> Union[Dict[str, Union[str, Dict]], RequestError]:
        """Fetch the current state of the job referenced with `comparison_id`. If the job has not yet been processed, it will return a `status` of `PENDING`. If the job is being processed by
            the service at the time of return, `status` will read `IN_PROGRESS`. If the job is complete, the job state will be returned, optionally with included result data.

            Args:
                comparison_id:`str`: The id of the comparison job submission.

            Returns:
                The job state of the task referenced by `comparison_id`. If the job has not yet been processed, it will return a `status` of `PENDING`.
        """
        piece = f'fetch-results/{comparison_id}'
        endpoint = self._format_endpoint(piece)

        headers = {'Accept': 'application/json'}

        try:
            response = requests.get(endpoint, headers=headers)
            self._check_response(response)
            return response.json()
        except Exception as e:
            return RequestError(error=str(e))

    def visualize(self):
        # TODO: get results and viz here
        pass

    def _format_endpoint(self, path_piece: str) -> str:
        return f'{self.endpoint_root}/{path_piece}'

    def _check_response(self, resp: Response) -> None:
        if resp.status_code != 200:
            raise Exception(f"Request failed:\n{resp.status_code}\n{resp.text}\n")
    
    def _test_root(self) -> dict:
        try:
            resp = requests.get(self.endpoint_root)
            resp.raise_for_status()
            return resp.json()
        except RequestException as e:
            return {'bio-check-error': f"A connection to that endpoint could not be established: {e}"}

