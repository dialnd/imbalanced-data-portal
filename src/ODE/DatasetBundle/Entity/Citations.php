<?php

namespace ODE\DatasetBundle\Entity;

use Doctrine\ORM\Mapping as ORM;

/**
 * Citations
 *
 * @ORM\Table(name="ode_citations")
 * @ORM\Entity(repositoryClass="ODE\DatasetBundle\Entity\CitationsRepository")
 */
class Citations
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
     * @var array
     *
     * @ORM\Column(name="citations", type="array")
     */
    private $citations;

    /**
     * @var integer
     *
     * @ORM\Column(name="dataset_id", type="integer")
     */
    private $datasetId;


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
     * Set citations
     *
     * @param array $citations
     * @return Citations
     */
    public function setCitations($citations)
    {
        $this->citations = $citations;

        return $this;
    }

    /**
     * Get citations
     *
     * @return array
     */
    public function getCitations()
    {
        return $this->citations;
    }

    /**
     * Set datasetId
     *
     * @param integer $datasetId
     * @return Keywords
     */
    public function setDatasetId($datasetId)
    {
        $this->datasetId = $datasetId;

        return $this;
    }

    /**
     * Get datasetId
     *
     * @return integer
     */
    public function getDatasetId()
    {
        return $this->datasetId;
    }
}
