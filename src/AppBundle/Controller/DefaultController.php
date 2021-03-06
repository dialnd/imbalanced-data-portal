<?php

namespace AppBundle\Controller;

use Sensio\Bundle\FrameworkExtraBundle\Configuration\Route;
use Symfony\Bundle\FrameworkBundle\Controller\Controller;

class DefaultController extends Controller
{
    public function indexAction()
    {
        // Checks whether the user has seen the homepage, ensures a new user is greeted only a single time after login/registration.
        // TODO: Debug this further
        if (!isset($_SESSION['greet'])){ $_SESSION['greet'] = 0; }

        $user = $this->getUser();
        if ($user) {
            $_SESSION['greet']++;
        } else {
            $_SESSION['greet'] = 0;
        }

        return $this->render('default/index.html.twig', array('greet' => $_SESSION['greet']));
    }
}
