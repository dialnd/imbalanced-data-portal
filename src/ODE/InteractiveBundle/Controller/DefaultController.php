<?php

namespace ODE\InteractiveBundle\Controller;

use Symfony\Bundle\FrameworkBundle\Controller\Controller;
use Symfony\Component\HttpFoundation\Request;
use Symfony\Component\HttpFoundation\Response;

class DefaultController extends Controller
{
    public function indexAction($name)
    {
        return $this->render('ODEInteractiveBundle:Default:index.html.twig', array('name' => $name));
    }

    /**
     * Loads the interface to choose or upload a dataset.
     */
    public function createDataset(Request $request) {

    }

    /**
     * Loads the interface to perform under/oversampling.
     */
    public function createSampling(Request $request) {

    }

    /**
     * Updates the visualization of under/oversampling.
     */
    public function updateSampling(Request $request) {

    }

    /**
     * Loads the interface to perform preprocessing.
     */
    public function createPreprocess(Request $request) {

    }

    /**
     * Updates the visualization of preprocessing.
     */
    public function updatePreprocess(Request $request) {

    }

    /**
     * Loads the interface to perform dimensionality reduction.
     */
    public function createReduction(Request $request) {

    }

    /**
     * Updates the visualization of dimensionality reduction.
     */
    public function updateReduction(Request $request) {

    }

    /**
     * Loads the interface to choose a model and parameterization.
     */
    public function createModel(Request $request) {

    }

    /**
     * Creates the visualization of model analysis.
     */
    public function createAnalysis(Request $request) {

    }
}
