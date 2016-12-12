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
