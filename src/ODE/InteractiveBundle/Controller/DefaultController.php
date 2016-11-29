<?php

namespace ODE\InteractiveBundle\Controller;

use Symfony\Bundle\FrameworkBundle\Controller\Controller;
use Symfony\Component\HttpFoundation\Request;
use Symfony\Component\HttpFoundation\Response;

class DefaultController extends Controller
{
    // This should be a constant static variable, but such functionality is
    // not compatible with many versions of PHP.
    private $stages = array(
        'workflow' => 0,
        'dataset' => 1,
        'preprocessing' => 2,
        'reduction' => 3,
        'sampling' => 4,
        'model' => 5,
        'parameterization' => 6,
        'results' => 7
    );

    private $templates = array(
        'workflow' => 'ode_workflow_page',
        'dataset' => 'ode_dataset_page',
        'sampling' => 'ode_sampling_page',
        'preprocessing' => 'ode_preprocessing_page',
        'reduction' => 'ode_reduction_page',
        'model' => 'ode_model_page',
        'parameterization' => 'ode_parameterization_page',
        'results' => 'ode_results_page'
    );

    public function indexAction($stage, Request $request) {
        $session = $this->get('session');
        $workflow = array();

        if ($session->has('workflow')){
            $workflow = $session->get('workflow', $workflow);
        }
        else {
            return $this->redirect($this->generateUrl($this->templates['dataset']));
        }

        $destination = $this->determineStage($stage);
        return $this->redirect($this->generateUrl($this->templates[$destination]));
    }

    public function createWorkflowAction(Request $request) {
        $session = $this->get('session');
        $workflow = null;

        if (!$session->has('workflow')) {
            $session->set('workflow', array());
        }

        return $this->render();
    }

    /**
     * Loads the interface to choose or upload a dataset.
     */
    public function createDatasetAction(Request $request) {
        $session = $this->get('session');
        $em = $this->getDoctrine()->getManager();
        $datasets = $em->getRepository('ODEDatasetBundle:Dataset')->findBy([], ['name' => 'ASC']);
        $dataset = null;

        if ($session->has('workflow')) {
            $workflow = $session->get('workflow');
            if (array_key_exists('dataset', $workflow)) {
                $dataset = $workflow['dataset'];
            }
        }

        return $this->render('@ODEInteractive/Default/createDataset.html.twig',
                             array(
                                'datasets' => $datasets,
                                'current' => $dataset
                             )
                            );
    }

    public function submitDatasetAction(Request $request) {
        $session = $this->get('session');
        $dataset_id = $request->request('dataset_id');

        $em = $this->getDoctrine()->getManager();
        $datasets = $em->getRepository('ODEDatasetBundle:Dataset')->findBy([], ['name' => 'ASC']);

        $workflow = array();
        if ($session->has('workflow')) {
            $workflow = $session->get('workflow');
        }

        $workflow['dataset'] = $dataset_id;

        $session->set("workflow", $workflow);

        return $this->redirect('/interactive/preprocessing');
    }

    /**
     * Loads the interface to perform under/oversampling.
     */
    public function createSamplingAction(Request $request) {
        return $this->render('@ODEInteractive/Default/createSampling.html.twig');
    }

    /**
     * Updates the visualization of under/oversampling.
     */
    public function updateSamplingAction(Request $request) {
        $session = $this->get('session');

        $workflow = array();

        if ($session->has('workflow')) {
            $workflow = $session->get('workflow');
        }
        else {
            $error = array('msg' => 'Error: No active workflow.');
            $response = new Response(json_encode($images));
            $response->headers->set('Content-Type', 'application/json');
            return $response;
        }

        $w_id = $workflow['id'];

        $undersampling = ($request->request->get('undersampling') == '1' ? true : false);
        $undersampling_rate = $request->request->get('undersampling_rate');
        $oversampling = ($request->request->get('oversampling') == '1' ? true : false);
        $oversampling_percentage = $request->request->get('oversampling_percentage');

        $script = 'run_sampling.py';
        $script_args = "--workflow $w_id";

        if ($undersampling) {
            $script_args .= " --undersample --undersample-rate $undersampling_rate"
        }

        if ($oversampling) {
            $script_args .= " --oversample --oversample-rate $oversampling_percentage"
        }

        $this->runPython($script, $script_args);


    }

    public function submitSamplingAction(Request $request) {
        $session = $this->get('session');
        $workflow = array();

        if ($session->has('workflow')) {
            $workflow = $session->get('workflow');
        }
        else {
            $this->redirect('/interactive/dataset');
        }

        $undersampling = ($request->request->get('undersampling') == '1' ? true : false);
        $undersampling_rate = $request->request->get('undersampling_rate');
        $oversampling = ($request->request->get('oversampling') == '1' ? true : false);
        $oversampling_percentage = $request->request->get('oversampling_percentage');

        $sampling = array();
        $sampling['undersampling'] = $undersampling;
        $sampling['undersampling_rate'] = $undersampling_rate;
        $sampling['oversampling'] = $oversampling;
        $sampling['oversampling_percentage'] = $oversampling_percentage;

        $workflow['sampling'] = $subsampling_rate;
        $session->set('workflow', $workflow);

        return $this->redirect('/interactive/model');
    }

    /**
     * Loads the interface to perform preprocessing.
     */
    public function createPreprocessingAction(Request $request) {
        return $this->render('@ODEInteractive/Default/createPreprocessing.html.twig');
    }

    /**
     * Updates the visualization of preprocessing.
     */
    public function updatePreprocessingAction(Request $request) {
        $session = $this->get("session");
        $workflow = $session->get("workflow");
        $dataset = $workflow["dataset"];
        $w_id = $workflow["id"];

        $missing_data = $request->request->get("missing_data");
        $standardization = ($request->request->get("standardization") == '1' ? true : false);
        $normalization = ($request->request->get("normalization") == '1' ? true : false);
        $norm_method = $request->request->get("norm");
        $binarization = ($request->request->get("binarization") == '1' ? true : false);
        $b_thresh = $request->request->get("binarization_threshold");
        $outliers = $request->request->get("outlier_detection");

        $script_args = "--dataset $dataset --workflow $w_id --missing-data $missing_data";

        if ($standardization) {
            $script_args .= " --standardization";
        }

        if ($normalization) {
            $script_args .= " --normalization --norm-method $norm_method";
        }

        if ($binarization) {
            $script_args .= " --binarization --binarization-threshold $b_thresh";
        }

        $this->runPython("run_preprocessing.py", $script_args);

        $response = new Response(json_encode($images));
        $response->headers->set('Content-Type', 'application/json');
        return $response;
    }

    public function submitPreprocessingAction(Request $request) {
        $session = $this->get("session");

        if ($session->has('workflow')) {
            $workflow = $session->get("workflow");
        }

        $settings = array();
        $settings["missing_data"] = $request->request->get("missing_data");
        $settings["standardization"] = ($request->request->get("standardization") == '1' ? true : false);
        $settings["normalization"] = ($request->request->get("normalization") == '1' ? true : false);
        $settings["norm_method"] = $request->request->get("norm");
        $settings["binarization"] = $request->request->get("binarization");
        $settings["binarization_threshold"] = $request->request->get("binarization_threshold");

        $workflow["preprocessing"] = $preprocessing;
        $session->set("workflow", $workflow);

        return $this->redirect($this->generateUrl($this->templates["reduction"]));
    }

    /**
     * Loads the interface to perform dimensionality reduction.
     */
    public function createReductionAction(Request $request) {
        return $this->render('@ODEInteractive/Default/createReduction.html.twig');
    }

    /**
     * Updates the visualization of dimensionality reduction.
     */
    public function updateReductionAction(Request $request) {
        $session = $this->get('session');
        $workflow = array();

        if ($session->has('workflow')) {
            $workflow = $session->get('workflow');

            if (!array_key_exists('id', $workflow)) {
                $this->redirect('/interactive/workflow');
            }
        }
        else {
            $this->redirect('/interactive/workflow');
        }

        $w_id = $workflow['id'];

        $reduction = ($request->query->get('pca') == '1' ? true : false);
        $n_components = $request->query->get('n_componenets');

        $script = 'run_reduction.py';
        $script_args = "--workflow $w_id";

        if ($reduction) {
            $script_args .= "--reduce --n-components $n_components";
        }

        $this->runPython($script, $script_args);


    }

    public function submitReductionAction(Request $request) {
        $session = $this->get('session');
        $workflow = array();
        $stage = $this->determineStage('reduction');

        if (!strcmp($stage, 'reduction')) {
            $workflow = $session->get('workflow');
        }
        else {
            return $this->redirect($this->generateUrl($this->templates[$destination]));
        }

        $reduction = array();
        $reduction['pca'] = $request->request->get('pca');
        $reduction['n_components'] = $request->request->get('n_components');

        $workflow['reduction'] = $reduction;
        $session->set('workflow', $workflow);

        $this->redirect('/interactive/sampling');
    }

    /**
     * Loads the interface to choose a model and parameterization.
     */
    public function createModelAction(Request $request) {
        $em = $this->getDoctrine()->getManager();
        $models = $em->getRepository('ODEAnalysisBundle:Model')->findBy([], ['name' => 'ASC']);
        return $this->render('@ODEInteractive/Default/createModel.html.twig',
                             array(
                                 'models' => $models
                             ));

    }

    public function submitModelAction(Request $request) {
        $session = $this->get('session');
        $workflow = array();
        $stage = $this->determineStage('model');

        if (!strcmp($stage, 'model')) {
            $workflow = $session->get('workflow');
        }
        else {
            return $this->redirect($this->generateUrl($this->templates[$destination]));
        }

        $workflow['model'] = $request->request->get('model');

        $session->set('workflow', $workflow);

        return $this->redirect('/interactive/parameterization');
    }

    /**
     * Loads the interface to choose a model and parameterization.
     */
    public function createParameterizationAction(Request $request) {
        $session = $this->get('session');
        $workflow = $session->get('workflow');

        if (array_key_exists('dataset', $workflow) && array_key_exists('model', $workflow)) {
            return $this->render('@ODEInteractive/Default/createParameterization.html.twig',
                                 array(
                                     'dataset' => $workflow['dataset'],
                                     'model' => $workflow['model']
                                 ));
        }
        else {
            return $this->redirect($this->generateUrl($this->templates['dataset']));
        }
    }

    public function updateParameterizationAction(Request $request) {

    }

    /**
     * Creates the visualization of model analysis.
     */
    public function createAnalysisAction(Request $request) {
        return $this->render('@ODEInteractive/Default/createAnalysis.html.twig');
    }

    /**
     * Ensure that the stages are being followed in order.
     *
     * Because the stages must occur in a specific order, maintaining that order
     * is key. This function allows enforcing the order and guarding against
     * failures due to bad routing.
     */
    private function determineStage($stage) {
        $session = $this->get('session');
        $stage_num = -1;

        if (array_key_exists($stage, $this->stages)) {
            $stage_num = $this->stages[$stage];
        }

        //
        if ($session->has('workflow') && $stage_num > 0) {
            $workflow = $session->get('workflow');

            if (!array_key_exists('dataset', $workflow) && $stage_num >= $this->stages["dataset"]) {
                return 'dataset';
            }
            else if (!array_key_exists('sampling', $workflow) && $stage_num >= $this->stages["sampling"]) {
                return 'sampling';
            }
            else if (!array_key_exists('preprocessing', $workflow) && $stage_num >= $this->stages["preprocessing"]) {
                return 'preprocessing';
            }
            else if (!array_key_exists('reduction', $workflow) && $stage_num >= $this->stages["reduction"]) {
                return 'reduction';
            }
            else if (!array_key_exists('model', $workflow) && $stage_num >= $this->stages["model"]) {
                return 'model';
            }
            else if (!array_key_exists('parameterization', $workflow) && $stage_num >= $this->stages["parameterization"]) {
                return 'parameterization';
            }
            else {
                return $stage;
            }

        }
        else {
            return 'workflow';
        }
    }

    private function runPython($script, $script_args) {
        // Python script needs to know the "current_dir" to open data files
        $script = __DIR__.'/../../../../web/assets/scripts/'.$script;
        $current_dir = __DIR__.'/../../../../web/assets/scripts/';

        if (substr(php_uname(), 0, 7) == "Windows"){
            $terminal_output = pclose(popen('start /B python '.$script.' '.$script_args, "r"));
        }  else {
            $terminal_output = exec('python '.$script.' '.$script_args.' &');
        }
    }

    private function saveWorkflow($workflow) {

    }
}
