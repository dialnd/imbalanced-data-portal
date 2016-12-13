<?php

namespace ODE\InteractiveBundle\Controller;

use Symfony\Bundle\FrameworkBundle\Controller\Controller;
use Symfony\Component\HttpFoundation\Request;
use Symfony\Component\HttpFoundation\Response;
use ODE\InteractiveBundle\Entity\Workflow;
use Symfony\Component\Validator\Constraints\DateTime;

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

    private $model_map = array(
        "Decision Tree" => "decision_tree",
        "Gaussian Naive Bayes" => "naive_bayes",
        "K-Nearest Neighbors" => "knn",
        "Logistic Regression" => "logistic_regression",
        "Support Vector Machine" => "svc",
        "Stochastic Gradient Descent" => "sgd",
        "Random Forest" => "random_forest",
        "Extremely Randomized Trees" => "extra_trees",
        "Gradient Boosting" => "gradient_boost"
    );

    public function indexAction($stage, Request $request) {
        $session = $this->get('session');
        $workflow = array();

        if ($session->has('workflow')){
            $workflow = $session->get('workflow');
        }
        else {
            return $this->redirect($this->generateUrl($this->templates['workflow']));
        }

        $destination = $this->determineStage($stage);
        return $this->redirect($this->generateUrl($this->templates[$destination]));
    }

    public function createWorkflowAction(Request $request) {
        $session = $this->get('session');
        $workflow = null;

        $user = $this->getUser();
        $em = $this->getDoctrine()->getManager();
        $workflows = $em->getRepository('ODEInteractiveBundle:Workflow')->findBy(['user' => $user->getId()], ['name' => 'ASC']);

        #if ($session->has('workflow')) {
        #    $workflow = $em->getRepository('ODEInteractiveBundle:Workflow')->find($session->get('workflow'));
        #}

        return $this->render('@ODEInteractive/Default/createWorkflow.html.twig',
                             array(
                                 'workflows' => $workflows,
                                 'current' => $workflow
                             )
        );
    }

    public function submitWorkflowAction(Request $request) {
        $session = $this->get('session');
        $w_id = $request->request->get('workflow');

        if ($w_id === 'new') {
            $em = $this->getDoctrine()->getManager();
            $name = $request->request->get('name');
            $workflow = new Workflow();
            $workflow->setName($name);
            $workflow->setUser($this->getUser());
            $workflow->setDate(new \DateTime());
            $workflow->setPreprocessing(array());
            $workflow->setReduction(array());
            $workflow->setSampling(array());
            $workflow->setParameterization(array());
            $workflow->setResult(array());
            $workflow->setMaxStage('workflow');
            $em->persist($workflow);
            $em->flush();
            $session->set('workflow', $workflow->getId());
        }
        else {
            $session->set('workflow', $w_id);
        }

        return $this->redirect('/interactive/dataset');
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
            $workflow = $em->getRepository('ODEInteractiveBundle:Workflow')->find($session->get('workflow'));
            $dataset = $workflow->getDataset();
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
        $dataset_id = $request->request->get('dataset');

        $em = $this->getDoctrine()->getManager();
        $dataset = $em->getRepository('ODEDatasetBundle:Dataset')->find($dataset_id);

        $workflow = $em->getRepository('ODEInteractiveBundle:Workflow')->find($session->get('workflow'));
        $workflow->setDataset($dataset);
        $workflow->setMaxStage('dataset');
        $em->flush();

        return $this->redirect('/interactive/preprocessing');
    }

    /**
     * Loads the interface to perform under/oversampling.
     */
    public function createSamplingAction(Request $request) {
        $session = $this->get('session');
        $em = $this->getDoctrine()->getManager();
        $workflow = $em->getRepository('ODEInteractiveBundle:Workflow')->find($session->get('workflow'));
        $reduction = $workflow->getReduction();
        $dataset = $workflow->getDataset();

        $n_features = $dataset->getNumFeatures();

        if ($reduction['pca']) {
            $n_features = (int)$reduction['n_components'];
        }

        return $this->render('@ODEInteractive/Default/createSampling.html.twig',
                             array(
                                 "n_features" => $n_features
                             )
        );
    }

    /**
     * Updates the visualization of under/oversampling.
     */
    public function updateSamplingAction(Request $request) {
        $session = $this->get('session');
        $w_id = $session->get('workflow');

        $undersampling = ($request->request->get('undersampling') == '1' ? true : false);
        $undersampling_rate = $request->request->get('undersampling_rate');
        $oversampling = ($request->request->get('oversampling') == '1' ? true : false);
        $oversampling_percentage = $request->request->get('oversampling_percentage');

        $script = 'run_sampling.py';
        $script_args = "--workflow $w_id";

        if ($undersampling) {
            $script_args .= " --undersample --undersample-rate $undersampling_rate";
        }

        if ($oversampling) {
            $script_args .= " --oversample --oversample-rate $oversampling_percentage";
        }

        $this->runPython($script, $script_args);

        $graphs = file_get_contents(__DIR__.'/../../../../web/assets/workflows/'.$w_id.'/sampling/graph_data.json');

        $response = new Response($graphs);
        $response->setStatusCode(200);
        $response->headers->set('Content-Type', 'application/json');
        $response->headers->set( 'X-Status-Code', 200 );
        return $response;
    }

    public function submitSamplingAction(Request $request) {
        $session = $this->get('session');
        $em = $this->getDoctrine()->getManager();
        $workflow = $em->getRepository('ODEInteractiveBundle:Workflow')->find($session->get('workflow'));

        $undersampling = ($request->request->get('undersampling') == '1' ? true : false);
        $undersampling_rate = $request->request->get('undersampling_rate');
        $oversampling = ($request->request->get('oversampling') == '1' ? true : false);
        $oversampling_percentage = $request->request->get('oversampling_percentage');

        $sampling = array();
        $sampling['undersampling'] = $undersampling;
        $sampling['undersampling_rate'] = $undersampling_rate;
        $sampling['oversampling'] = $oversampling;
        $sampling['oversampling_percentage'] = $oversampling_percentage;

        $workflow->setSampling($sampling);
        $workflow->setMaxStage('sampling');
        $em->flush();

        return $this->redirect('/interactive/model');
    }

    /**
     * Loads the interface to perform preprocessing.
     */
    public function createPreprocessingAction(Request $request) {
        $session = $this->get('session');
        $em = $this->getDoctrine()->getManager();
        $workflow = $em->getRepository('ODEInteractiveBundle:Workflow')->find($session->get('workflow'));
        $dataset = $workflow->getDataset();
        $n_features = $dataset->getNumFeatures();
        $w_id = $session->get('workflow');
        if (!file_exists(__DIR__.'/../../../../web/assets/scripts/../workflows/'.$w_id)) {
            mkdir(__DIR__.'/../../../../web/assets/scripts/../workflows/'.$w_id, 0777, true);
            mkdir(__DIR__.'/../../../../web/assets/scripts/../workflows/'.$w_id.'/preprocessing');
            mkdir(__DIR__.'/../../../../web/assets/scripts/../workflows/'.$w_id.'/reduction');
            mkdir(__DIR__.'/../../../../web/assets/scripts/../workflows/'.$w_id.'/fold0');
            mkdir(__DIR__.'/../../../../web/assets/scripts/../workflows/'.$w_id.'/fold1');
            mkdir(__DIR__.'/../../../../web/assets/scripts/../workflows/'.$w_id.'/fold2');
            mkdir(__DIR__.'/../../../../web/assets/scripts/../workflows/'.$w_id.'/fold3');
            mkdir(__DIR__.'/../../../../web/assets/scripts/../workflows/'.$w_id.'/fold4');
            mkdir(__DIR__.'/../../../../web/assets/scripts/../workflows/'.$w_id.'/fold5');
            mkdir(__DIR__.'/../../../../web/assets/scripts/../workflows/'.$w_id.'/fold6');
            mkdir(__DIR__.'/../../../../web/assets/scripts/../workflows/'.$w_id.'/fold7');
            mkdir(__DIR__.'/../../../../web/assets/scripts/../workflows/'.$w_id.'/fold8');
            mkdir(__DIR__.'/../../../../web/assets/scripts/../workflows/'.$w_id.'/fold9');
            mkdir(__DIR__.'/../../../../web/assets/scripts/../workflows/'.$w_id.'/parameterization');
            mkdir(__DIR__.'/../../../../web/assets/scripts/../workflows/'.$w_id.'/sampling');
        }

        if (!$n_features || $n_features < 0) {
            $n_features = 4;
        }

        return $this->render('@ODEInteractive/Default/createPreprocessing.html.twig',
                             array(
                                 'n_features' => $n_features
                             )
                        );
    }

    /**
     * Updates the visualization of preprocessing.
     */
    public function updatePreprocessingAction(Request $request) {
        $session = $this->get("session");
        $w_id = $session->get("workflow");
        $em = $this->getDoctrine()->getManager();
        $workflow = $em->getRepository('ODEInteractiveBundle:Workflow')->find($session->get('workflow'));
        $dataset = $workflow->getDataset();

        $missing_data = $request->query->get("missing_data");
        $standardization = ($request->query->get("standardization") == '1' ? true : false);
        $normalization = ($request->query->get("normalization") == '1' ? true : false);
        $norm_method = $request->query->get("norm");
        $binarization = ($request->query->get("binarization") == '1' ? true : false);
        $b_thresh = $request->query->get("binarization_threshold");
        $outliers = ($request->query->get("outlier_detection") == '1' ? true : false);


        $filename = $dataset->getFilename();
        $script_args = "--dataset $filename.csv --workflow $w_id --missing-data $missing_data";

        if ($standardization) {
            $script_args .= " --standardization";
        }

        if ($normalization) {
            $script_args .= " --normalization --norm-method $norm_method";
        }

        if ($binarization) {
            $script_args .= " --binarization --binarization-threshold $b_thresh";
        }

        $output = $this->runPython("run_preprocessing.py", $script_args);

        $graphs = file_get_contents(__DIR__.'/../../../../web/assets/scripts/../workflows/'.$w_id.'/preprocessing/graph_data.json');

        $response = new Response($graphs);
        $response->setStatusCode(200);
        $response->headers->set('Content-Type', 'application/json');
        $response->headers->set( 'X-Status-Code', 200 );
        return $response;
    }

    public function submitPreprocessingAction(Request $request) {
        $session = $this->get("session");
        $em = $this->getDoctrine()->getManager();
        $workflow = $em->getRepository('ODEInteractiveBundle:Workflow')->find($session->get('workflow'));

        $settings = array();
        $settings["missing_data"] = $request->request->get("missing_data");
        $settings["standardization"] = ($request->request->get("standardization") == '1' ? true : false);
        $settings["normalization"] = ($request->request->get("normalization") == '1' ? true : false);
        $settings["norm_method"] = $request->request->get("norm");
        $settings["binarization"] = $request->request->get("binarization");
        $settings["binarization_threshold"] = $request->request->get("binarization_threshold");

        $workflow->setPreprocessing($settings);
        $workflow->setMaxStage('preprocessing');
        $em->flush();

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
        $w_id = $session->get('workflow');

        $reduction = ($request->query->get('pca') == '1' ? true : false);
        $n_components = $request->query->get('n_components');

        $script = 'run_reduction.py';
        $script_args = "--workflow $w_id";

        if ($reduction) {
            $script_args .= " --reduce --n-components $n_components";
        }

        $this->runPython($script, $script_args);

        $graphs = file_get_contents(__DIR__.'/../../../../web/assets/workflows/'.$w_id.'/reduction/graph_data.json');

        $response = new Response($graphs);
        $response->setStatusCode(200);
        $response->headers->set('Content-Type', 'application/json');
        $response->headers->set( 'X-Status-Code', 200 );
        return $response;
    }

    public function submitReductionAction(Request $request) {
        $session = $this->get('session');
        $em = $this->getDoctrine()->getManager();
        $workflow = $em->getRepository('ODEInteractiveBundle:Workflow')->find($session->get('workflow'));

        $reduction = array();
        $reduction['pca'] = $request->query->get('pca');
        $reduction['n_components'] = $request->query->get('n_components');

        $workflow->setReduction($reduction);
        $workflow->setMaxStage('reduction');
        $em->flush();

        return $this->redirect('/interactive/sampling');
    }

    /**
     * Loads the interface to choose a model and parameterization.
     */
    public function createModelAction(Request $request) {
        $session = $this->get('session');
        $em = $this->getDoctrine()->getManager();
        $models = $em->getRepository('ODEAnalysisBundle:Model')->findBy([], ['name' => 'ASC']);
        $workflow = $em->getRepository('ODEInteractiveBundle:Workflow')->find($session->get('workflow'));
        $model = $workflow->getModel();
        return $this->render('@ODEInteractive/Default/createModel.html.twig',
                             array(
                                 'models' => $models,
                                 'model' => $model
                             ));

    }

    public function submitModelAction(Request $request) {
        $session = $this->get('session');
        $em = $this->getDoctrine()->getManager();
        $workflow = $em->getRepository('ODEInteractiveBundle:Workflow')->find($session->get('workflow'));
        $model = $workflow->getModel();

        $model_id = $request->query->get('model');

        if (!$model || $model_id != $model->getID()) {
            $model = $em->getRepository('ODEAnalysisBundle:Model')->find($model_id);
            $workflow->setModel($model);
            $workflow->setParameterization(array());
        }

        $workflow->setMaxStage('model');
        $em->flush();

        return $this->redirect('/interactive/parameterization');
    }

    /**
     * Loads the interface to choose a model and parameterization.
     */
    public function createParameterizationAction(Request $request) {
        $session = $this->get('session');
        $em = $this->getDoctrine()->getManager();
        $workflow = $em->getRepository('ODEInteractiveBundle:Workflow')->find($session->get('workflow'));
        $dataset = $workflow->getDataset();
        $model = $workflow->getModel();

        return $this->render('@ODEInteractive/Default/createParameterization.html.twig',
                     array(
                         'dataset' => $dataset,
                         'model' => $model
                     ));
    }

    public function updateParameterizationAction(Request $request) {
        $session = $this->get('session');
        $w_id = $session->get('workflow');
        $em = $this->getDoctrine()->getManager();
        $workflow = $em->getRepository('ODEInteractiveBundle:Workflow')->find($session->get('workflow'));
        $model = $workflow->getModel();

        $model_params = $model->getParameters();
        $params = array();
        foreach ($model_params as $key => $value) {
            $model_params->{$key} = $request->query->get($key);
        }

        $name = $this->model_map[$model->getName()];
        $params = json_encode($model_params);
        $script = 'run_model.py';
        $script_args = "-w $w_id --model $name --model-params $model_params";

        $this->runPython($script, $script_args);

        $graphs = file_get_contents(__DIR__.'/../../../../web/assets/scripts/../workflows/'.$w_id.'/parameterization/graph_data.json');

        $response = new Response($graphs);
        $response->setStatusCode(200);
        $response->headers->set('Content-Type', 'application/json');
        $response->headers->set( 'X-Status-Code', 200 );
        return $response;
    }

    public function submitParameterizationAction(Request $request) {
        $session = $this->get('session');
        $em = $this->getDoctrine()->getManager();
        $workflow = $em->getRepository('ODEInteractiveBundle:Workflow')->find($session->get('workflow'));
        $dataset = $workflow->getDataset();
        $model = $workflow->getModel();

        $model_params = $model->getParameters();
        $params = array();
        foreach ($model_params as $key => $value) {
            $params[$key] = $request->get($key);
        }

        $workflow->setParameterization($params);
        $workflow->setMaxStage('parameterization');
        $em->flush();

        return $this->redirect('/interactive/results');
    }

    /**
     * Creates the visualization of model analysis.
     */
    public function createAnalysisAction(Request $request) {
        $session = $this->get('session');
        $w_id = $session->get('workflow');

        $em = $this->getDoctrine()->getManager();
        $workflow = $em->getRepository('ODEInteractiveBundle:Workflow')->find($session->get('workflow'));
        $dataset = $workflow->getDataset();
        $model = $workflow->getModel();
        $user = $workflow->getUser();

        $model_name = $this->model_map[$model->getName()];
        $model_params = json_encode($workflow->getParameterization());
        file_put_contents(__DIR__.'/../../../../web/assets/scripts/../workflows/'.$w_id.'/parameterization.json', $model_params);
        #$model_params =

        $script = 'run_model.py';
        $script_args = "-w $w_id --analysis --model $model_name --model-params parameterization.json";

        $this->runPython($script, $script_args);

        $res = json_decode(file_get_contents(__DIR__.'/../../../../web/assets/scripts/../workflows/'.$w_id.'/results.json'));
        $result = array();
        foreach ($res as $key => $value) {
            $result[$key] = $value;
        }

        $workflow->setResult($result);
        $workflow->setMaxStage('results');
        $em->flush();

        return $this->render('@ODEInteractive/Default/createResult.html.twig',
            array(
                'auroc' => $result['auroc'],
                'aupr' => $result['aupr'],
                'roc_points' => $result['roc_points'],
                'prc_points' => $result['prc_points'],
                'confusion_matrix' => explode(",",$result['confusion_matrix']),
                'classification_report' => explode(",",$result['classification_report']),
                'indexes' => explode(",",$result['indexes']),
                'y_original_values' => explode(",",$result['y_original_values']),
                'y_pred' => explode(",",$result['y_pred']),
                'errors' => explode(",",$result['errors']),
                'y_prob' => explode(",",$result['y_prob']),
                'model' => $model->getName(),
                'dataset_name' => $dataset->getName(),
                'runtime' => $result['runtime'],
                'params' => $model_params,
                'accuracy' => $result['accuracy'],
                'user' => $user->getUsername(),
                'rank' => 1
            )
        );
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
        $max_stage_num = -1;
        $stage_num = -1;

        if (array_key_exists($stage, $this->stages)) {
            $stage_num = $this->stages[$stage];
        }

        //
        if ($session->has('workflow') && $stage_num > 0) {
            $em = $this->getDoctrine()->getManager();
            $workflow = $em->getRepository('ODEInteractiveBundle:Workflow')->find($session->get('workflow'));
            if (array_key_exists($workflow->getMaxStage(), $this->stages)) {
                $max_stage_num = $this->stages[$workflow->getMaxStage()];
            }

            if (!$workflow->getDataset() && $stage_num >= $this->stages["dataset"] && $max_stage_num >= $this->stages["dataset"]) {
                return 'dataset';
            }
            else if ($workflow->getPreprocessing() === array() && $stage_num >= $this->stages["preprocessing"] && $max_stage_num >= $this->stages["preprocessing"]) {
                return 'preprocessing';
            }
            else if ($workflow->getReduction() === array() && $stage_num >= $this->stages["reduction"] && $max_stage_num >= $this->stages["reduction"]) {
                return 'reduction';
            }
            else if ($workflow->getSampling() === array() && $stage_num >= $this->stages["sampling"] && $max_stage_num >= $this->stages["sampling"]) {
                return 'sampling';
            }
            else if (!$workflow->getModel() && $stage_num >= $this->stages["model"] && $max_stage_num >= $this->stages["model"]) {
                return 'model';
            }
            else if ($workflow->getParameterization() === array() && $stage_num >= $this->stages["parameterization"] && $max_stage_num >= $this->stages["parameterization"]) {
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

    private function runToMax($stage, Workflow $workflow) {
        $stage_num = -1;
        $max_stage_num = -1;
        $max_stage = $workflow->getMaxStage();

        $w_id = $workflow->getId();
        $dataset = $workflow->getDataset();
        $preprocessing = $workflow->getPreprocessing();
        $reduction = $workflow->getReduction();
        $sampling = $workflow->getSampling();
        $model = $workflow->getModel();
        $parameterization = $workflow->getParameterization();

        if (array_key_exists($stage, $this->stages)) {
            $stage_num = $this->stages[$stage];
        }

        if (array_key_exists($max_stage, $this->stages)) {
            $max_stage_num = $this->stages[$max_stage];
        }

        if ($stage_num < $this->stages["preprocessing"] || $max_stage_num < $this->stages["preprocessing"]) {
            return;
        }

        if ($stage_num <= $this->stages["preprocessing"] && $max_stage_num >= $this->stages["preprocessing"]) {
            $script = 'run_preprocessing.py';
            $filename = $dataset->getFilename();
            $missing_data = $preprocessing['missing_data'];
            $script_args = "--dataset $filename.csv --workflow $w_id --missing-data $missing_data";

            if ($preprocessing['standardization']) {
                $script_args .= " --standardization";
            }

            if ($preprocessing['normalization']) {
                $norm_method = $preprocessing['norm_method'];
                $script_args .= " --normalization --norm-method $norm_method";
            }

            if ($preprocessing['binarization']) {
                $b_thresh = $preprocessing['binarization_threshold'];
                $script_args .= " --binarization --binarization-threshold $b_thresh";
            }

            $this->runPython($script, $script_args);
        }

        if ($stage_num <= $this->stages["reduction"] && $max_stage_num >= $this->stages["reduction"]) {
            $script = 'run_reduction.py';
            $script_args = "--workflow $w_id";

            if ($reduction['reduction']) {
                $n_components = $reduction['n_components'];
                $script_args .= "--reduce --n-components $n_components";
            }

            $this->runPython($script, $script_args);
        }

        if ($stage_num <= $this->stages["sampling"] && $max_stage_num >= $this->stages["sampling"]) {
            $script = 'run_sampling.py';
            $script_args = "--workflow $w_id";


            if ($sampling['undersampling']) {
                $undersampling_rate = $sampling['undersampling_rate'];
                $script_args .= " --undersample --undersample-rate $undersampling_rate";
            }

            if ($sampling['oversampling']) {
                $oversampling_percentage = $sampling['oversampling_percentage'];
                $script_args .= " --oversample --oversample-rate $oversampling_percentage";
            }

            $this->runPython($script, $script_args);
        }
    }

    private function runPython($script, $script_args) {
        // Python script needs to know the "current_dir" to open data files
        $script = __DIR__.'/../../../../web/assets/scripts/'.$script;
        $current_dir = __DIR__.'/../../../../web/assets/scripts/';
        $script_args .= ' -c '.$current_dir;

        if (substr(php_uname(), 0, 7) == "Windows"){
            $terminal_output = pclose(popen('start /B python '.$script.' '.$script_args, "r"));
        }  else {
            $terminal_output = exec('python '.$script.' '.$script_args.' &');
        }

        return $terminal_output;
    }

    private function saveWorkflow($workflow) {

    }
}
