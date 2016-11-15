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
        'sampling' => 2,
        'preprocessing' => 3,
        'reduction' => 4,
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
        $workflow = array(
            'workflow' => 0,
            'dataset' => 1,
            'sampling' => 2,
            'preprocessing' => 3,
            'reduction' => 4,
            'model_selection' => 5,
            'parameterization' => 6,
            'results' => 7
        );
        $session = $this->get('session');
        $session->set('workflow', $workflow);

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

        $workflow = array();
        if ($session->has('workflow')) {
            $workflow = $session->get('workflow');
        }

        $workflow['dataset'] = $dataset_id;

        return $this->redirect($this->generateUrl('ode_wait_result', array('id' => $result->getId()),true));
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

    private function saveWorkflow($workflow) {

    }
}
