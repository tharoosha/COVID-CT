# with torch.no_grad():
#     for batch_index, batch_samples in enumerate(train_loader):
#       data, target = batch_samples['img'].to(device), batch_samples['label'].to(device)

#       # Forward pass
#       try:
#         outputs = model(data)
#         pred = torch.round(torch.squeeze(torch.sigmoid(outputs)))
#         # print(data.shape)
#         loss = criterion(pred, target.float())
#         print("current testing loss :", loss)
#         test_loss += loss

#         for t, p in zip(target.view(-1), pred.view(-1)):
#           # print([t.long(), p.long()])
#           confusion_matrix_train[t.long(), p.long()] += 1
                    

#       except Exception as e:
#         print(f"An error occurred: {e}")

#     precision_train, recall_train, f1_train, accuracy_train = getMetrics(confusion_matrix_train)
#     print("Training accuracy: {} {} {} {}".format(accuracy_train, precision_train, recall_train, f1_train, ))
#     training_loss = train_loss / len(test_loader)
#     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}'.format(epoch, batch_index, len(train_loader),100.0 * batch_index / len(train_loader), loss.item()/ 1))


from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# For each epoch, initialize metric accumulators to 0
total_precision, total_recall, total_f1, total_accuracy = 0, 0, 0, 0

with torch.no_grad():
    for batch_index, batch_samples in enumerate(train_loader):
        data, target = batch_samples['img'].to(device), batch_samples['label'].to(device)

        # Forward pass
        outputs = model(data)
        
        

        # Convert outputs to predicted class
        _, predicted = torch.max(outputs.data, 1)
        
        # Calculate loss
        loss = criterion(predicted.float(), target.float())

        # Calculate metrics for this batch
        target_cpu = target.cpu().numpy()
        predicted_cpu = predicted.cpu().numpy()
        
        batch_precision = precision_score(target_cpu, predicted_cpu)
        batch_recall = recall_score(target_cpu, predicted_cpu)
        batch_f1 = f1_score(target_cpu, predicted_cpu)
        batch_accuracy = accuracy_score(target_cpu, predicted_cpu)
        
        total_precision += batch_precision
        total_recall += batch_recall
        total_f1 += batch_f1
        total_accuracy += batch_accuracy

    # Average the metrics across batches for the epoch
    epoch_precision = total_precision / 17
    epoch_recall = total_recall / 17
    epoch_f1 = total_f1 / 17
    epoch_accuracy = total_accuracy / 17
    
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Precision: {epoch_precision:.4f} | Recall: {epoch_recall:.4f} | F1 Score: {epoch_f1:.4f} | Accuracy: {epoch_accuracy:.4f}")
