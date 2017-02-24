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
     * @ORM\ManyToOne(targetEntity="ODE\DatasetBundle\Entity\Dataset", inversedBy="keywords")
     * @ORM\JoinColumn(name="dataset_id", referencedColumnName="id")
     */
    private $dataset;


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


    public function setDataset($dataset)
    {
        $this->dataset = $dataset;

        return $this;
    }

    public function getDataset()
    {
        return $this->dataset;
    }
}
