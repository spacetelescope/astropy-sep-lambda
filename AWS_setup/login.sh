#!/usr/bin/env bash

ec2_ip=$1  # first arg is IP address of instance
aws_dir=~/.aws/
aws_ssh_key=$aws_dir.aws_ssh/ndmiles_admin_useast1.pem

# log into the instance
ssh -i "$aws_ssh_key" ec2-user@$ec2_ip