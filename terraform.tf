terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.43"
    }
  }

  required_version = ">= 1.15"
}

