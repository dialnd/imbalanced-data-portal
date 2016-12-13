<?php

namespace ODE\InteractiveBundle\Entity;

use Doctrine\ORM\Mapping as ORM;

/**
 * Workflow
 *
 * @ORM\Table()
 * @ORM\Entity(repositoryClass="ODE\InteractiveBundle\Entity\WorkflowRepository")
 */
class Workflow
{
    /**
     * @var integer
     *
     * @ORM\Column(name="id", type="integer")
     * @ORM\Id
     * @ORM\GeneratedValue(strategy="AUTO")
     */
    private $id;

    /**
     * @var string
     *
     * @ORM\Column(name="name", type="string", length=255)
     */
    private $name;

    /**
     * @ORM\ManyToOne(targetEntity="ODE\UserBundle\Entity\User", inversedBy="workflow")
     * @ORM\JoinColumn(name="user_id", referencedColumnName="id")
     */
    private $user;

    /**
     * @var \DateTime
     *
     * @ORM\Column(name="date", type="datetime")
     */
    private $date;

    /**
     * @ORM\ManyToOne(targetEntity="ODE\AnalysisBundle\Entity\Model", inversedBy="workflow")
     * @ORM\JoinColumn(name="model_id", referencedColumnName="id")
     */
    private $model;

    /**
     * @ORM\ManyToOne(targetEntity="ODE\DatasetBundle\Entity\Dataset", inversedBy="workflow")
     * @ORM\JoinColumn(name="dataset_id", referencedColumnName="id")
     */
    private $dataset;

    /**
     * @var array
     *
     * @ORM\Column(name="preprocessing", type="json_array")
     */
    private $preprocessing;

    /**
     * @var array
     *
     * @ORM\Column(name="reduction", type="json_array")
     */
    private $reduction;

    /**
     * @var array
     *
     * @ORM\Column(name="sampling", type="json_array")
     */
    private $sampling;

    /**
     * @var array
     *
     * @ORM\Column(name="parameterization", type="json_array")
     */
    private $parameterization;

    /**
     * @var array
     *
     * @ORM\Column(name="result", type="json_array")
     */
    private $result;

    /**
     * @var string
     *
     * @ORM\Column(name="max_stage", type="string", length=255)
     */
    private $maxStage;


    public function doStuffOnPrePersist()
    {
        $this->date = date('Y-m-d H:i:s');
    }

    /**
     * Get id
     *
     * @return integer
     */
    public function getId()
    {
        return $this->id;
    }

    /**
     * Set maxStage
     *
     * @param string $maxStage
     * @return Workflow
     */
    public function setName($name)
    {
        $this->name = $name;
    }

    /**
     * Get maxStage
     *
     * @return string
     */
    public function getName()
    {
        return $this->name;
    }

    public function setUser($user)
    {
        $this->user = $user;

        return $this;
    }

    public function getUser()
    {
        return $this->user;
    }

    /**
     * Set date
     *
     * @param \DateTime $date
     * @return Workflow
     */
    public function setDate($date)
    {
        $this->date = $date;

        return $this;
    }

    /**
     * Get date
     *
     * @return \DateTime
     */
    public function getDate()
    {
        return $this->date;
    }

    public function setModel($model)
    {
        $this->model = $model;

        return $this;
    }

    public function getModel()
    {
        return $this->model;
    }

    public function setDataset($dataset)
    {
        $this->dataset = $dataset;

        return $this;
    }

    public function getDataset()
    {
        return $this->dataset;
    }

    /**
     * Set preprocessing
     *
     * @param array $preprocessing
     * @return Workflow
     */
    public function setPreprocessing($preprocessing)
    {
        $this->preprocessing = $preprocessing;

        return $this;
    }

    /**
     * Get preprocessing
     *
     * @return array
     */
    public function getPreprocessing()
    {
        return $this->preprocessing;
    }

    /**
     * Set reduction
     *
     * @param array $reduction
     * @return Workflow
     */
    public function setReduction($reduction)
    {
        $this->reduction = $reduction;

        return $this;
    }

    /**
     * Get reduction
     *
     * @return array
     */
    public function getReduction()
    {
        return $this->reduction;
    }

    /**
     * Set sampling
     *
     * @param array $sampling
     * @return Workflow
     */
    public function setSampling($sampling)
    {
        $this->sampling = $sampling;

        return $this;
    }

    /**
     * Get sampling
     *
     * @return array
     */
    public function getSampling()
    {
        return $this->sampling;
    }

    /**
     * Set parameterization
     *
     * @param array $parameterization
     * @return Workflow
     */
    public function setParameterization($parameterization)
    {
        $this->parameterization = $parameterization;

        return $this;
    }

    /**
     * Get parameterization
     *
     * @return array
     */
    public function getParameterization()
    {
        return $this->parameterization;
    }

    /**
     * Set result
     *
     * @param array $result
     * @return Workflow
     */
    public function setResult($result)
    {
        $this->result = $result;

        return $this;
    }

    /**
     * Get result
     *
     * @return array
     */
    public function getResult()
    {
        return $this->result;
    }

    /**
     * Set maxStage
     *
     * @param string $maxStage
     * @return Workflow
     */
    public function setMaxStage($maxStage)
    {
        $this->maxStage = $maxStage;

        return $this;
    }

    /**
     * Get maxStage
     *
     * @return string
     */
    public function getMaxStage()
    {
        return $this->maxStage;
    }
}
