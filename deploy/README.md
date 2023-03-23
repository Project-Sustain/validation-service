# Deploying Validation Service

## Prerequisites

The nodes you want a worker instance running on, i.e. that are holding data and running 
mongod instances need to be labeled as `validationSvcWorker=true` using:
```bash
kubectl label node <node_name> validationSvcWorker=true
```

This label allows a Kubernetes [DaemonSet](https://kubernetes.io/docs/concepts/workloads/controllers/daemonset/) 
to launch a worker pod _only_ on the nodes labeled as such.

## Deployment Steps

1. Deploy your master pod:
    ```bash
    kubectl apply -f pod_master.yaml
    ```
2. Take note of the IP address assigned to the pod (IP field):
    ```bash
    kubectl get pod validation-svc-master -o yaml | grep "podIP:"
    ```
3. Edit [worker_daemonset.yaml](./worker_daemonset.yaml) and replace the
the `PodSpec.containers.[0].command` field's master URI and port to the `<pod_ip>:50059`
   - Example: `command: [ "python3", "-m", "overlay", "--worker", "--master_uri=<master-ip>:<master-port>", "--port=50058" ]`
4. Deploy the DaemonSet once the master pod has fully spun up:
    ```bash
    kubectl apply -f worker_daemonset.yaml
    ```
5. Deploy Flask Server
   ```bash
    kubectl apply -f pod_flask.yaml
6. ```
6. Use `k9s` or `kubectl` to watch the logs of the master pod, or worker pods as they come online:
   - Master: `kubectl logs -f validation-svc-master -c validation-svc-master`
   - Flask: `kubectl logs -f validation-svc-flask -c validation-svc-flask`
   - `k9s` -> `:daemonsets`

## Teardown

Delete the resources you created in the reverse order:

```bash
kubectl delete -f worker_daemonset.yaml
kubectl delete -f pod_master.yaml
kubectl delete -f pod_flask.yaml
```