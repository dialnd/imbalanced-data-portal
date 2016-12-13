<?php

namespace ODE\DatasetBundle\Entity;

use Doctrine\ORM\Mapping as ORM;

/**
 * Keywords
 *
 * @ORM\Table(name="ode_keywords")
 * @ORM\Entity(repositoryClass="ODE\DatasetBundle\Entity\KeywordsRepository")
 */
class Keywords
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
     * @ORM\Column(name="keywords", type="array")
     */
    private $keywords;

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
     * Set keywords
     *
     * @param array $keywords
     * @return Keywords
     */
    public function setKeywords($keywords)
    {
        $this->keywords = $keywords;

        return $this;
    }

    /**
     * Get keywords
     *
     * @return array
     */
    public function getKeywords()
    {
        return $this->keywords;
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
