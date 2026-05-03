provider "aws" {
  region = "eu-north-1"
}

data "aws_ami" "amazon_linux" {
  most_recent = true

  filter {
    name   = "name"
    values = ["al2023-ami-2023.*-x86_64"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }

  owners = ["137112412989"] # Amazon
}

resource "aws_security_group" "ml_api" {
  name        = "ml-model-serving"
  description = "SSH access and public HTTPS for ML model API"

  tags = {
    Name = "ml-model-serving"
  }
}

resource "aws_vpc_security_group_ingress_rule" "ssh" {
  security_group_id = aws_security_group.ml_api.id
  cidr_ipv4         = "0.0.0.0/0"
  from_port         = 22
  ip_protocol       = "tcp"
  to_port           = 22
}

resource "aws_vpc_security_group_ingress_rule" "http" {
  security_group_id = aws_security_group.ml_api.id
  cidr_ipv4         = "0.0.0.0/0"
  from_port         = 80
  ip_protocol       = "tcp"
  to_port           = 80
}

resource "aws_vpc_security_group_egress_rule" "all_outbound" {
  security_group_id = aws_security_group.ml_api.id
  cidr_ipv4         = "0.0.0.0/0"
  ip_protocol       = "-1"
}

resource "aws_key_pair" "ml_api" {
  key_name = "ml-model-serving"
  public_key = file("~/.ssh/ml-model-serving.pub")
}


resource "aws_s3_bucket" "ml_models" {
  bucket_prefix = "ml-model-serving-"

  tags = {
    Name = "ml-model-serving"
  }
}

resource "aws_s3_bucket_public_access_block" "ml_models" {
  bucket = aws_s3_bucket.ml_models.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_iam_role" "ec2_s3_read" {
  name = "ml-model-serving-ec2"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "s3_read" {
  name = "ml-models-s3-read"
  role = aws_iam_role.ec2_s3_read.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "s3:GetObject",
        "s3:ListBucket"
      ]
      Resource = [
        aws_s3_bucket.ml_models.arn,
        "${aws_s3_bucket.ml_models.arn}/*"
      ]
    }]
  })
}

resource "aws_iam_instance_profile" "ec2_s3_read" {
  name = "ml-model-serving-ec2"
  role = aws_iam_role.ec2_s3_read.name
}

resource "aws_instance" "app_server" {
  ami                    = data.aws_ami.amazon_linux.id
  instance_type          = "t3.micro"
  vpc_security_group_ids = [aws_security_group.ml_api.id]
  key_name               = aws_key_pair.ml_api.key_name
  iam_instance_profile   = aws_iam_instance_profile.ec2_s3_read.name

  tags = {
    Name = "ml-model-serving"
  }
}

output "s3_bucket_name" {
  value = aws_s3_bucket.ml_models.bucket
}

output "instance_public_ip" {
  value = aws_instance.app_server.public_ip
}

