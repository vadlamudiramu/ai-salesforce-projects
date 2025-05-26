
// Apex Trigger - Contact Update on Case Creation
trigger CaseContactUpdater on Case (after insert) {
    Set<Id> contactIds = new Set<Id>();
    for (Case c : Trigger.new) {
        if (c.ContactId != null) {
            contactIds.add(c.ContactId);
        }
    }

    List<Contact> contactsToUpdate = [SELECT Id, LastCaseCreated__c FROM Contact WHERE Id IN :contactIds];
    for (Contact con : contactsToUpdate) {
        con.LastCaseCreated__c = System.now();
    }

    update contactsToUpdate;
}
